#!/usr/bin/env bash
# Creates the Python environment Voice Type needs and downloads the speech model.
set -euo pipefail

dir="${XDG_DATA_HOME:-$HOME/.local/share}/voice-type"
venv="$dir/venv"
model="${VOICE_TYPE_MODEL:-base.en}"
mkdir -p "$dir"

command -v pw-record >/dev/null || { echo "pw-record not found (it ships with PipeWire)." >&2; exit 1; }

# faster-whisper's dependencies need Python 3.9-3.12; uv fetches one if the system Python is newer.
if command -v uv >/dev/null; then
    uv venv --allow-existing --python 3.12 "$venv"
    uv pip install --python "$venv/bin/python" faster-whisper
else
    py=""
    for c in python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not (3, 9) <= sys.version_info[:2] <= (3, 12))'; then
            py="$c"; break
        fi
    done
    [ -n "$py" ] || { echo "Need Python 3.9-3.12, or install uv (https://docs.astral.sh/uv/)." >&2; exit 1; }
    "$py" -m venv "$venv"
    "$venv/bin/pip" install --upgrade pip faster-whisper
fi

echo "Downloading the $model speech model..."
"$venv/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('$model', device='cpu', compute_type='int8')"
echo "Done. Enable the extension with: gnome-extensions enable voice-type@arjun"
