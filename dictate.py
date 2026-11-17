#!/usr/bin/env python3
"""Live dictation helper for the voice-type@arjun GNOME extension.

Listens to the default mic and streams JSON lines on stdout while you talk:
  {"t": "ready"}                       models loaded
  {"t": "partial", "text": "..."}      words heard but not yet certain (preview only)
  {"t": "commit",  "text": "..."}      final text to type now (append-only, never revised)
  {"t": "done"}                        finished; the process exits after this
A line (or EOF) on stdin stops listening; it also stops after AUTO_STOP seconds
without speech.

How it streams: Silero VAD splits speech into utterances. While an utterance is
in progress it is re-transcribed every STEP seconds, and words that two
consecutive passes agree on are committed ("local agreement"). When a pause
ends the utterance, a final pass commits the rest.

Env: VOICE_TYPE_MODEL (base.en), VOICE_TYPE_THREADS (all cores), VOICE_TYPE_AUTO_STOP (30),
VOICE_TYPE_TEST_INPUT (raw s16le 16 kHz mono file played back in real time, for tests).
"""
import collections
import json
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata

import numpy as np

RATE = 16000
FRAME = 512                       # Silero frame, 32 ms
FRAME_S = FRAME / RATE
MODEL = os.environ.get("VOICE_TYPE_MODEL", "base.en")
THREADS = int(os.environ.get("VOICE_TYPE_THREADS", os.cpu_count() or 4))
AUTO_STOP = float(os.environ.get("VOICE_TYPE_AUTO_STOP", "30"))
TEST_INPUT = os.environ.get("VOICE_TYPE_TEST_INPUT")

START_PROB, END_PROB = 0.5, 0.35
END_SILENCE = 0.6                 # pause that ends an utterance
PREROLL = 0.3                     # audio kept from before speech was detected
STEP = 0.4                        # min new audio between live passes
MIN_LIVE = 0.8                    # no live commits until the utterance is this long
MAX_UTT = 20.0                    # force an utterance boundary
PROMPT_CHARS = 200
# Whisper's stock outputs for noise; only dropped when they are a whole short utterance.
HALLUCINATIONS = {"you", "thanks for watching", "thank you for watching",
                  "subtitles by the amaraorg community"}


def emit(kind, text=None):
    msg = {"t": kind} if text is None else {"t": kind, "text": text}
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


# The extension types through a virtual keyboard, which can only send keys
# that exist in the keymap, so fold typographic characters down to ASCII.
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                       "–": "-", "—": " - ", "…": "...", " ": " "})


def to_ascii(text):
    text = unicodedata.normalize("NFKD", text.translate(_FOLD))
    return text.encode("ascii", "ignore").decode()


def norm(word):
    return re.sub(r"[^\w']", "", word.lower())


class Mic:
    """Collects 16 kHz mono int16 audio in a background thread."""

    def __init__(self):
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.proc = None
        self.running = True
        if TEST_INPUT:
            threading.Thread(target=self._play_file, daemon=True).start()
        else:
            self.proc = subprocess.Popen(
                ["pw-record", "--raw", "--rate", str(RATE), "--channels", "1",
                 "--format", "s16", "--media-role", "Communication", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while chunk := self.proc.stdout.read(3200):
            with self.lock:
                self.buf.extend(chunk)
        self.running = False

    def _play_file(self):
        with open(TEST_INPUT, "rb") as f:
            data = f.read()
        i = 0
        while self.running:      # file in real time, then silence
            chunk = data[i:i + 3200] or bytes(3200)
            i += 3200
            with self.lock:
                self.buf.extend(chunk)
            time.sleep(0.1)

    def take(self):
        with self.lock:
            n = len(self.buf) // (FRAME * 2) * FRAME * 2
            raw = bytes(self.buf[:n])
            del self.buf[:n]
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Vad:
    """Streaming Silero VAD (the ONNX model bundled with faster-whisper)."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self.session = get_vad_model().session
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)

    def __call__(self, frame):
        inp = np.concatenate([self.ctx, frame])[None, :]
        out, self.h, self.c = self.session.run(None, {"input": inp, "h": self.h, "c": self.c})
        self.ctx = frame[-64:]
        return float(np.asarray(out).reshape(-1)[0])


class Dictation:
    def __init__(self, model):
        self.model = model
        self.history = ""        # text committed in earlier utterances (prompt context)
        self.typed_any = False
        self._new_utterance([])

    def _new_utterance(self, frames):
        self.frames = list(frames)
        self.committed = []      # words of this utterance already typed
        self.prev = None         # previous live hypothesis
        self.fresh = 0           # frames since the last live pass
        self.prompt = self.history[-PROMPT_CHARS:].strip() or None

    def _transcribe(self):
        audio = np.concatenate(self.frames)
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, temperature=0.0,
            without_timestamps=True, condition_on_previous_text=False,
            vad_filter=False, initial_prompt=self.prompt)
        words = []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
                continue
            words += [w for w in (to_ascii(x) for x in s.text.split()) if w.strip()]
        return words

    def _commit(self, words):
        if not words:
            return
        text = " ".join(words)
        emit("commit", (" " if self.typed_any else "") + text)
        self.typed_any = True
        self.committed += words

    def duration(self):
        return len(self.frames) * FRAME_S

    def add(self, frame):
        self.frames.append(frame)
        self.fresh += 1

    def live_pass(self):
        if self.fresh * FRAME_S < STEP or self.duration() < MIN_LIVE:
            return
        self.fresh = 0
        hyp = self._transcribe()
        n = len(self.committed)
        if self.prev is not None and [norm(w) for w in hyp[:n]] == [norm(w) for w in self.committed]:
            agree = 0
            for a, b in zip(self.prev, hyp):
                if a != b:
                    break
                agree += 1
            if agree > n:
                self._commit(hyp[n:agree])
        self.prev = hyp
        emit("partial", " ".join(hyp[len(self.committed):]))

    def finish(self, next_frames=()):
        if self.frames:
            hyp = self._transcribe()
            if (not self.committed and self.duration() < 1.5
                    and " ".join(norm(w) for w in hyp) in HALLUCINATIONS):
                hyp = []
            # Typed words are never revised; only words past them are added.
            self._commit(hyp[len(self.committed):])
            if self.committed:
                self.history = (self.history + " " + " ".join(self.committed))[-PROMPT_CHARS * 2:]
        emit("partial", "")
        self._new_utterance(next_frames)


def main():
    mic = Mic()
    stop = threading.Event()

    def wait_stdin():
        sys.stdin.readline()
        stop.set()
    threading.Thread(target=wait_stdin, daemon=True).start()

    from faster_whisper import WhisperModel
    try:  # use the cached model without touching the network; download only if missing
        model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS,
                             local_files_only=True)
    except Exception:
        model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=THREADS)
    vad = Vad()
    emit("ready")

    d = Dictation(model)
    preroll = collections.deque(maxlen=int(PREROLL / FRAME_S))
    in_speech = False
    silence = 0
    last_speech = time.monotonic()

    while True:
        stopping = stop.is_set() or not mic.running
        samples = mic.take()
        for frame in samples.reshape(-1, FRAME):
            p = vad(frame)
            if p >= START_PROB:
                last_speech = time.monotonic()
            if not in_speech:
                preroll.append(frame)
                if p >= START_PROB:
                    in_speech, silence = True, 0
                    d._new_utterance(preroll)
                    preroll.clear()
                continue
            d.add(frame)
            silence = silence + 1 if p < END_PROB else 0
            if silence * FRAME_S >= END_SILENCE:
                d.finish()
                in_speech = False
            elif d.duration() >= MAX_UTT:
                d.finish()
        if stopping:
            if in_speech:
                d.finish()
            break
        if in_speech:
            d.live_pass()
        if AUTO_STOP and not in_speech and time.monotonic() - last_speech > AUTO_STOP:
            break
        if samples.size == 0:
            time.sleep(0.03)

    mic.stop()
    if d.typed_any:
        emit("commit", " ")
    emit("done")


if __name__ == "__main__":
    main()
