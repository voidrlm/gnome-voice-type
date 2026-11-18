#!/usr/bin/env python3
"""Live dictation helper for the voice-type@arjun GNOME extension.

Listens to the default mic and streams JSON lines on stdout while you talk:
  {"t": "ready"}                       models loaded
  {"t": "partial", "text": "..."}      words heard but not yet certain (preview only)
  {"t": "commit",  "text": "..."}      final text to type now (append-only, never revised)
  {"t": "done"}                        finished; the process exits after this
A line (or EOF) on stdin stops listening; it also stops after AUTO_STOP seconds
without speech.

How it streams: audio goes into one rolling buffer that spans your pauses
(long silences are squeezed to KEEP_GAP), so Whisper always hears the whole
sentence and doesn't end it just because you stopped to think. The buffer is
re-transcribed with word timestamps every STEP seconds of new speech; words that
two passes in a row agree on are typed ("local agreement"), except the newest
word, whose punctuation depends on what follows. After a pause the held words
are typed but a trailing . , ? ! is held back; once you speak again it is typed
only if Whisper, now hearing what came next, still puts it there. The buffer is
trimmed at committed sentence ends.

Env: VOICE_TYPE_MODEL (base.en), VOICE_TYPE_THREADS (up to 8), VOICE_TYPE_AUTO_STOP (30),
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
from dataclasses import dataclass

import numpy as np

RATE = 16000
FRAME = 512                       # Silero frame, 32 ms
FRAME_S = FRAME / RATE
MODEL = os.environ.get("VOICE_TYPE_MODEL", "base.en")
# More than 8 threads doesn't speed up base.en and only competes with other apps.
THREADS = int(os.environ.get("VOICE_TYPE_THREADS", min(8, os.cpu_count() or 4)))
AUTO_STOP = float(os.environ.get("VOICE_TYPE_AUTO_STOP", "30"))
TEST_INPUT = os.environ.get("VOICE_TYPE_TEST_INPUT")

START_PROB, END_PROB = 0.5, 0.35  # VAD hysteresis
KEEP_GAP = 0.3                    # pauses longer than this are squeezed to this in the buffer
PREROLL = 0.2                     # audio kept from just before speech resumes
FLUSH_AFTER = 0.8                 # pause after which held-back words are typed
STEP = 0.4                        # new speech between live passes
TRIM_AT = 16.0                    # buffer length that triggers a trim (keeps passes under ~0.8 s)
KEEP_AFTER_TRIM = 9.0             # audio kept after a trim; heard context is what keeps base.en accurate
HARD_TRIM = 22.0                  # trim at any typed word if no sentence end turns up
MIN_SPEECH = 0.25                 # words lying mostly in non-speech audio are Whisper inventions
PROMPT_CHARS = 200
PUNCT = re.compile(r"[.,!?;:]+$")
ELLIPSIS = re.compile(r"(\.\.\.|\.\.)$")     # Whisper's mark for a hesitation
# Whisper's stock outputs for noise; dropped when they are all that was heard.
HALLUCINATIONS = {"you", "thank you", "thanks for watching", "thank you for watching",
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


@dataclass
class Word:
    text: str
    start: float   # seconds from the start of the buffer
    end: float
    soft: bool = False   # Whisper ended it with "..." (a hesitation, not a sentence end)
    glue: bool = False   # attaches to the previous word without a space ("12" + "%")
    cut: float = 0.0     # real end time, for trimming (end may be capped)


def is_cap_word(text):
    """Words that stay capitalized mid-sentence: I, I'm, I'll..., and acronyms."""
    w = re.sub(r"[^\w']", "", text)
    return w == "I" or w.startswith("I'") or (len(w) > 1 and w.isupper())


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
        self.frames = []          # rolling audio buffer
        self.speech = []          # VAD verdict per frame
        self.last_voiced = 0.0    # buffer time of the last clearly voiced frame
        self.context = ""         # committed text that has been trimmed out of the buffer
        self.committed = []       # committed Words still inside the buffer
        self.committed_end = 0.0  # buffer time up to which everything is typed
        self.prev = None          # uncommitted words of the previous live pass
        self.pending = None       # (punctuation, word norm, word start) held back at a pause
        self.typed_any = False
        self.fresh = 0            # speech frames since the last pass
        self.dirty = False        # speech since the last flush

    # ---- audio

    def add(self, frame, speech, voiced=False):
        self.frames.append(frame)
        self.speech.append(speech)
        if voiced:
            self.last_voiced = self.duration()
        if speech:
            self.fresh += 1
            self.dirty = True

    def duration(self):
        return len(self.frames) * FRAME_S

    def _speech_share(self, start, end):
        a = max(0, int((start - 0.1) / FRAME_S))
        b = min(len(self.speech), int((end + 0.1) / FRAME_S) + 1)
        return sum(self.speech[a:b]) / max(1, b - a)

    # ---- transcription

    def _transcribe(self):
        segments, _ = self.model.transcribe(
            np.concatenate(self.frames), language="en", beam_size=1, temperature=0.0,
            word_timestamps=True, condition_on_previous_text=False, vad_filter=False,
            initial_prompt=self.context[-PROMPT_CHARS:].strip() or None)
        words = []
        for s in segments:
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
                continue
            for i, w in enumerate(s.words or []):
                text = to_ascii(w.word).strip()
                if not text or w.end - w.start < 0.04 or w.start > self.last_voiced + 0.05:
                    continue   # zero-length words, or words after the voice stopped, are inventions
                if words and words[-1].soft and text[:1].isupper() and not is_cap_word(text):
                    text = text[0].lower() + text[1:]   # "to... Go" -> "to go"
                if self._speech_share(w.start, w.end) < MIN_SPEECH:
                    continue   # e.g. "Thanks." or "Bye." made up in the silence after a phrase
                soft = bool(ELLIPSIS.search(text))
                text = ELLIPSIS.sub("", text) or text
                glue = i > 0 and not w.word[:1].isspace() and bool(words)
                words.append(Word(text, float(w.start), float(w.end), soft, glue))
        return words

    def _uncommitted(self, words):
        """Words of this pass that come after what has been typed."""
        real = [k for k, w in enumerate(self.committed) if norm(w.text)]
        if real:
            # Anchor on the last typed word: its text and start time are stable between
            # passes, unlike end times, which stretch into pauses.
            last = self.committed[real[-1]]
            after = len(self.committed) - 1 - real[-1]   # symbol-only words typed after it
            hits = [i for i, w in enumerate(words)
                    if norm(w.text) == norm(last.text) and abs(w.start - last.start) < 0.6]
            if hits:
                j = min(hits, key=lambda i: abs(words[i].start - last.start))
                return words[j + 1 + after:]
        tail = [w for w in words if (w.start + w.end) / 2 > self.committed_end]
        # Fallback when the last word changed: drop a repeat of the last committed words.
        if tail and self.committed and tail[0].start < self.committed_end + 1.0:
            for k in range(min(5, len(tail), len(self.committed)), 0, -1):
                if [norm(w.text) for w in tail[:k]] == [norm(w.text) for w in self.committed[-k:]]:
                    return tail[k:]
        return tail

    # ---- typing

    def _resolve_pending(self, words, first_new):
        """Punctuation held at the last pause: keep it only if Whisper, now hearing what
        came next, still ends the sentence there (or starts the next word with a capital)."""
        punct, key, start = self.pending
        self.pending = None
        if first_new is None:
            return punct
        starts_sentence = first_new.text[:1].isupper() and not is_cap_word(first_new.text)
        again = next((w for w in words if norm(w.text) == key and abs(w.start - start) < 0.6), None)
        if again is not None:
            m = PUNCT.search(again.text)
            if m:
                return m.group(0)
        if starts_sentence:
            return punct if punct[-1:] in (".", "?", "!") else "."
        return ""

    def _commit(self, new, words, hold_punct):
        text = ""
        if self.pending:
            text += self._resolve_pending(words, new[0] if new else None)
            if text and self.committed:   # so trimming can find this sentence end
                last = self.committed[-1]
                self.committed[-1] = Word(last.text + text, last.start, last.end, cut=last.cut)
        pieces = []
        for i, w in enumerate(new):
            t = w.text
            if hold_punct and i == len(new) - 1:
                # Decide this word's punctuation once we hear what follows the pause.
                m = PUNCT.search(t)
                if m and m.start() > 0:
                    t = t[:m.start()]
                self.pending = (m.group(0) if m else "", norm(t), w.start)
            if i == 0 and text and text[-1] in ".?!" and t[:1].islower():
                t = t[0].upper() + t[1:]
            pieces.append(t)
            # Cap the end: Whisper stretches a word's end across a following pause.
            self.committed.append(Word(t, w.start, min(w.end, w.start + 0.5), cut=w.end))
        if new:
            for i, (w, t) in enumerate(zip(new, pieces)):
                if i or self.typed_any:
                    text += "" if w.glue else " "
                text += t
            self.committed_end = self.committed[-1].end
            self.typed_any = True
        if text:
            emit("commit", text)
        self._trim()

    def _trim(self):
        dur = self.duration()
        if dur < TRIM_AT or not self.committed:
            return
        # Cut at a typed sentence end, leaving about KEEP_AFTER_TRIM seconds of audio.
        target = dur - KEEP_AFTER_TRIM
        ends = [w.cut for w in self.committed if re.search(r"[.!?]$", w.text)]
        if not ends and dur >= HARD_TRIM:
            ends = [w.cut for w in self.committed]
        if not ends:
            return
        cut = min(ends, key=lambda e: abs(e - target))
        n = int(cut / FRAME_S)
        shift = n * FRAME_S
        del self.frames[:n]
        del self.speech[:n]
        self.last_voiced -= shift
        gone = [w for w in self.committed if w.cut <= cut]
        self.context = (self.context + " " + " ".join(w.text for w in gone))[-PROMPT_CHARS * 2:]
        self.committed = [Word(w.text, w.start - shift, w.end - shift, cut=w.cut - shift)
                          for w in self.committed if w.cut > cut]
        self.committed_end -= shift
        if self.pending:
            p, key, start = self.pending
            self.pending = (p, key, start - shift)
        self.prev = None

    # ---- passes

    def live_pass(self):
        if self.fresh * FRAME_S < STEP:
            return
        self.fresh = 0
        words = self._transcribe()
        tail = self._uncommitted(words)
        agree = 0
        if self.prev is not None:
            for a, b in zip(self.prev, tail):
                if norm(a.text) != norm(b.text):
                    break
                agree += 1
        agree = min(agree, len(tail) - 1)   # hold back the newest word
        if agree > 0:
            self._commit(tail[:agree], words, hold_punct=False)
            tail = tail[agree:]
        self.prev = tail
        emit("partial", " ".join(w.text for w in tail))

    def flush(self, final=False):
        """Type everything heard so far (at a pause, or when stopping)."""
        if not self.frames or (not self.dirty and not final):
            if final and self.pending:
                self._commit([], [], hold_punct=False)
            return
        self.dirty = False
        self.fresh = 0
        words = self._transcribe()
        tail = self._uncommitted(words)
        if tail and not self.committed and " ".join(norm(w.text) for w in tail) in HALLUCINATIONS:
            tail = []
            self.committed_end = self.duration()
        if tail or (final and self.pending):
            self._commit(tail, words, hold_punct=not final)
        self.prev = None
        emit("partial", "")


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
    preroll = collections.deque(maxlen=max(1, int(PREROLL / FRAME_S)))
    talking = False
    gap = 10 ** 6                 # frames since speech; huge = nothing heard yet
    last_speech = time.monotonic()

    while True:
        stopping = stop.is_set() or not mic.running
        samples = mic.take()
        for frame in samples.reshape(-1, FRAME):
            p = vad(frame)
            talking = p >= (END_PROB if talking else START_PROB)
            if talking:
                if gap * FRAME_S > KEEP_GAP:      # resuming after a squeezed pause
                    for f in preroll:     # the voice detector reacts a little late
                        d.add(f, speech=True)
                preroll.clear()
                d.add(frame, speech=True, voiced=p >= START_PROB)
                gap = 0
                last_speech = time.monotonic()
            else:
                gap += 1
                if gap * FRAME_S <= KEEP_GAP:
                    d.add(frame, speech=False)
                else:
                    preroll.append(frame)
                if gap == round(FLUSH_AFTER / FRAME_S):
                    d.flush()
        if stopping:
            d.flush(final=True)
            break
        if talking or gap * FRAME_S < FLUSH_AFTER:
            d.live_pass()
        if AUTO_STOP and time.monotonic() - last_speech > AUTO_STOP:
            d.flush(final=True)
            break
        if samples.size == 0:
            time.sleep(0.03)

    mic.stop()
    if d.typed_any:
        emit("commit", " ")
    emit("done")


if __name__ == "__main__":
    main()
