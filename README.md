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

`dictate.py` records from the default mic and uses Silero VAD to find phrases. While you talk, the current phrase is re-transcribed every 0.4 s, and words that two passes in a row agree on are typed. Typed words are never revised; a pause ends the phrase and a final pass types the rest. The extension types through a virtual keyboard, so text is plain ASCII.

## Options

Set these in `~/.config/environment.d/voice-type.conf` and log back in:

| Variable | Default | |
|---|---|---|
| `VOICE_TYPE_MODEL` | `base.en` | `small.en` is more accurate but lags a few seconds on a laptop CPU |
| `VOICE_TYPE_AUTO_STOP` | `30` | seconds of silence before it stops (0 = never) |
| `VOICE_TYPE_THREADS` | all cores | CPU threads for Whisper |

Errors from the last session are in `~/.cache/voice-type.log`.
