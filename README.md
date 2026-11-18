# Voice Type

Windows-style voice typing for GNOME. Press **Super+V** and start talking: your words are typed into whatever window has focus as you speak, about a second behind your voice. A small pill under the top bar shows words that are still being worked out. Press **Super+V** again (or click the pill) to stop; it also stops after 30 seconds of silence.

Speech is transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper); nothing leaves your machine.

GNOME Shell extension (48–51). Needs PipeWire (`pw-record`) and Python 3.9–3.12 or [uv](https://docs.astral.sh/uv/).

## Install

```sh
git clone https://github.com/voidrlm/gnome-voice-type.git ~/.local/share/gnome-shell/extensions/voice-type@arjun
~/.local/share/gnome-shell/extensions/voice-type@arjun/setup.sh
gsettings set org.gnome.shell.keybindings toggle-message-tray "['<Super>m']"   # frees Super+V
gnome-extensions enable voice-type@arjun
```

Log out and back in (Wayland) for GNOME to discover it.

`setup.sh` creates a venv in `~/.local/share/voice-type/venv` and downloads the `base.en` model (~150 MB).

## How it works

`dictate.py` records from the default mic and keeps one rolling audio buffer that spans your pauses (long silences are squeezed to 0.3 s), so Whisper always hears the whole sentence. While you talk, the buffer is re-transcribed every 0.4 s and words that two passes in a row agree on are typed; typed words are never revised. Pausing doesn't end the sentence: after a pause the last word is typed but its full stop is held back until you speak again, and kept only if Whisper, hearing what came next, still ends the sentence there. Silero VAD skips silence (no work while you're quiet) and filters out words Whisper invents in silence ("Thanks.", "Bye."). The extension types through a virtual keyboard, so text is plain ASCII.

Nothing runs while you're not dictating: the helper starts on Super+V and exits when you stop. While listening it uses ~400 MB of RAM and a few CPU cores only while you speak.

## Options

Set these in `~/.config/environment.d/voice-type.conf` and log back in:

| Variable | Default | |
|---|---|---|
| `VOICE_TYPE_MODEL` | `base.en` | `small.en` is more accurate but lags a few seconds on a laptop CPU |
| `VOICE_TYPE_AUTO_STOP` | `30` | seconds of silence before it stops (0 = never) |
| `VOICE_TYPE_THREADS` | up to 8 | CPU threads for Whisper |

Errors from the last session are in `~/.cache/voice-type.log`.
