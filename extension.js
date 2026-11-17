import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Meta from 'gi://Meta';
import Pango from 'gi://Pango';
import Shell from 'gi://Shell';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const ACCELERATOR = '<Super>v';
// setup.sh creates this venv (faster-whisper); dictate.py ships with the extension.
const PYTHON = GLib.build_filenamev([GLib.get_user_data_dir(), 'voice-type', 'venv', 'bin', 'python']);
const LOG = GLib.build_filenamev([GLib.get_user_cache_dir(), 'voice-type.log']);

const HELD_MODS = Clutter.ModifierType.SUPER_MASK | Clutter.ModifierType.MOD4_MASK |
    Clutter.ModifierType.CONTROL_MASK | Clutter.ModifierType.MOD1_MASK;
const CHARS_PER_TICK = 6;

// Super+V toggles live dictation: dictate.py streams committed words as you
// talk and they are typed into the focused window through a virtual keyboard.
// A pill under the top bar previews words that are not final yet.
export default class VoiceTypeExtension extends Extension {
    enable() {
        this._state = 'idle';           // idle | starting | listening | finishing
        this._session = null;           // {proc, cancellable}
        this._queue = '';
        this._typeId = 0;

        const seat = global.stage.context.get_backend().get_default_seat();
        this._vkbd = seat.create_virtual_device(Clutter.InputDeviceType.KEYBOARD_DEVICE);

        this._buildPill();

        this._action = global.display.grab_accelerator(ACCELERATOR, Meta.KeyBindingFlags.NONE);
        if (this._action === Meta.KeyBindingAction.NONE) {
            Main.notify(this.metadata.name, `Could not grab ${ACCELERATOR}; another shortcut is using it.`);
        } else {
            Main.wm.allowKeybinding(Meta.external_binding_name_for_action(this._action),
                Shell.ActionMode.NORMAL | Shell.ActionMode.OVERVIEW | Shell.ActionMode.POPUP);
            this._accelId = global.display.connect('accelerator-activated', (_d, action) => {
                if (action === this._action)
                    this._toggle();
            });
        }
    }

    disable() {
        this._cancel();
        if (this._accelId)
            global.display.disconnect(this._accelId);
        this._accelId = 0;
        if (this._action && this._action !== Meta.KeyBindingAction.NONE) {
            global.display.ungrab_accelerator(this._action);
            Main.wm.allowKeybinding(Meta.external_binding_name_for_action(this._action), Shell.ActionMode.NONE);
        }
        this._action = 0;
        Main.layoutManager.removeChrome(this._pill);
        this._pill.destroy();
        this._pill = this._mic = this._label = null;
        this._vkbd = null;
    }

    // ---------- pill

    _buildPill() {
        this._pill = new St.BoxLayout({
            style_class: 'voice-type-pill',
            reactive: true,
            track_hover: true,
            visible: false,
        });
        this._mic = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'voice-type-mic',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._label = new St.Label({style_class: 'voice-type-text', y_align: Clutter.ActorAlign.CENTER});
        this._label.clutter_text.ellipsize = Pango.EllipsizeMode.START;
        this._pill.add_child(this._mic);
        this._pill.add_child(this._label);
        this._pill.connect('button-press-event', () => {
            this._toggle();
            return Clutter.EVENT_STOP;
        });
        this._pill.connect('notify::width', () => this._place());
        Main.layoutManager.addTopChrome(this._pill);
    }

    _place() {
        const mon = Main.layoutManager.primaryMonitor;
        if (!mon || !this._pill)
            return;
        this._pill.set_position(
            Math.floor(mon.x + (mon.width - this._pill.width) / 2),
            mon.y + Main.panel.height + 10);
    }

    _show(text, partial = false) {
        this._label.text = text;
        this._pill.visible = true;
        for (const s of ['starting', 'listening', 'finishing'])
            this._pill.remove_style_class_name(`voice-type-${s}`);
        this._pill.add_style_class_name(`voice-type-${this._state}`);
        if (partial)
            this._pill.add_style_class_name('voice-type-partial');
        else
            this._pill.remove_style_class_name('voice-type-partial');
        this._place();
    }

    _maybeHide() {
        if (this._state === 'idle' && !this._queue && this._pill)
            this._pill.visible = false;
    }

    // ---------- dictation process

    _toggle() {
        if (this._state === 'idle')
            this._start();
        else if (this._state === 'starting' || this._state === 'listening')
            this._stop();
    }

    _start() {
        const script = GLib.build_filenamev([this.path, 'dictate.py']);
        if (!GLib.file_test(PYTHON, GLib.FileTest.IS_EXECUTABLE)) {
            Main.notify(this.metadata.name, `Missing ${PYTHON}. Run setup.sh from ${this.path}.`);
            return;
        }
        let proc;
        try {
            GLib.file_set_contents(LOG, '');   // the launcher appends; keep only this run's log
            const launcher = new Gio.SubprocessLauncher({
                flags: Gio.SubprocessFlags.STDIN_PIPE | Gio.SubprocessFlags.STDOUT_PIPE,
            });
            launcher.set_stderr_file_path(LOG);
            proc = launcher.spawnv([PYTHON, script]);
        } catch (e) {
            Main.notify(this.metadata.name, `Could not start dictation: ${e.message}`);
            return;
        }
        const session = {proc, cancellable: new Gio.Cancellable()};
        this._session = session;
        this._state = 'starting';
        this._show('Starting…');

        const stdout = new Gio.DataInputStream({
            base_stream: proc.get_stdout_pipe(),
            close_base_stream: true,
        });
        this._readLine(session, stdout);
    }

    _readLine(session, stream) {
        stream.read_line_async(GLib.PRIORITY_DEFAULT, session.cancellable, (_s, res) => {
            let line = null;
            try {
                [line] = stream.read_line_finish_utf8(res);
            } catch (e) {
                if (e.matches(Gio.IOErrorEnum, Gio.IOErrorEnum.CANCELLED))
                    return;
            }
            if (line === null) {
                this._ended(session);
                return;
            }
            try {
                this._onMessage(session, JSON.parse(line));
            } catch (e) {
                console.warn(`voice-type: bad line ${line}: ${e}`);
            }
            this._readLine(session, stream);
        });
    }

    _onMessage(session, msg) {
        if (msg.t === 'commit') {
            this._enqueue(msg.text);   // type even if the session is already finishing
            return;
        }
        if (this._session !== session)
            return;
        if (msg.t === 'ready' && this._state === 'starting') {
            this._state = 'listening';
            this._show('Listening…');
        } else if (msg.t === 'partial' && this._state === 'listening') {
            this._show(msg.text || 'Listening…', !!msg.text);
        }
    }

    _ended(session) {
        session.proc.wait_async(null, () => {
            if (!session.proc.get_successful() && !session.cancellable.is_cancelled())
                Main.notify(this.metadata.name, `Dictation stopped with an error; see ${LOG}`);
        });
        if (this._session !== session)
            return;
        this._session = null;
        this._state = 'idle';
        this._maybeHide();
    }

    _stop() {
        // Closing stdin tells dictate.py to finish the current phrase and exit.
        this._state = 'finishing';
        this._show('…');
        try {
            this._session.proc.get_stdin_pipe().close(null);
        } catch (e) {
            this._session.proc.force_exit();
        }
    }

    _cancel() {
        if (this._session) {
            this._session.cancellable.cancel();
            this._session.proc.force_exit();
            this._session = null;
        }
        if (this._typeId)
            GLib.source_remove(this._typeId);
        this._typeId = 0;
        this._queue = '';
        this._state = 'idle';
        this._maybeHide();
    }

    // ---------- typing through a virtual keyboard

    _enqueue(text) {
        if (!text)
            return;
        this._queue += text;
        if (this._typeId)
            return;
        this._typeId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 8, () => {
            // Wait out a held Super/Ctrl/Alt (e.g. from Super+V) so text doesn't become shortcuts.
            const [, , mods] = global.get_pointer();
            if (mods & HELD_MODS)
                return GLib.SOURCE_CONTINUE;
            const chars = [...this._queue];
            for (const ch of chars.slice(0, CHARS_PER_TICK))
                this._sendChar(ch);
            this._queue = chars.slice(CHARS_PER_TICK).join('');
            if (this._queue)
                return GLib.SOURCE_CONTINUE;
            this._typeId = 0;
            this._maybeHide();
            return GLib.SOURCE_REMOVE;
        });
    }

    _sendChar(ch) {
        let keyval;
        if (ch === '\n')
            keyval = Clutter.KEY_Return;
        else if (ch === '\t')
            keyval = Clutter.KEY_Tab;
        else
            keyval = Clutter.unicode_to_keysym(ch.codePointAt(0));
        const t = GLib.get_monotonic_time();
        this._vkbd.notify_keyval(t, keyval, Clutter.KeyState.PRESSED);
        this._vkbd.notify_keyval(t, keyval, Clutter.KeyState.RELEASED);
    }
}
