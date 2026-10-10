from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "static" / "sounds"
RNG = np.random.default_rng(2026)


def finish(x: np.ndarray, peak: float = 0.72) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    if not len(x):
        return x.astype(np.float32)
    attack = min(len(x), int(SR * 0.004))
    release = min(len(x), int(SR * 0.035))
    if attack:
        x[:attack] *= np.linspace(0.0, 1.0, attack)
    if release:
        x[-release:] *= np.linspace(1.0, 0.0, release)
    x = np.tanh(x * 1.25)
    m = np.max(np.abs(x))
    if m:
        x *= peak / m
    return x.astype(np.float32)


def chirp(duration: float, f0: float, f1: float, decay: float = 7.0) -> np.ndarray:
    n = max(1, int(SR * duration))
    t = np.arange(n) / SR
    k = np.log(max(f1, 1.0) / max(f0, 1.0)) / max(duration, 1e-6)
    freq = f0 * np.exp(k * t)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return finish((np.sin(phase) + 0.18 * np.sin(2.03 * phase)) * np.exp(-decay * t))


def arpeggio(notes, step=0.08, tail=0.12, peak=0.60):
    total = int(SR * (len(notes) * step + tail))
    out = np.zeros(total)
    for i, f in enumerate(notes):
        start = int(i * step * SR)
        t = np.arange(int((step + tail) * SR)) / SR
        voice = (np.sin(2 * np.pi * f * t) + 0.18 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 12)
        end = min(total, start + len(voice))
        out[start:end] += voice[: end - start]
    return finish(out, peak)


def noise_hit(duration=0.16):
    t = np.arange(int(SR * duration)) / SR
    noise = RNG.normal(0, 1, len(t))
    tone = np.sin(2 * np.pi * 780 * t)
    return finish((0.55 * noise + tone) * np.exp(-t * 27), 0.62)


def super_blast(duration=0.75):
    t = np.arange(int(SR * duration)) / SR
    base = chirp(duration, 1300, 60, decay=4.8).astype(np.float64)
    sub = np.sin(2 * np.pi * 58 * t) * np.exp(-t * 5)
    return finish(base + 0.55 * sub, 0.88)


SOUNDS = {
    "blaster.wav": lambda: chirp(0.26, 2200, 120, 9.5),
    "charge.wav": lambda: chirp(0.20, 180, 980, 4.0),
    "super.wav": super_blast,
    "hit.wav": noise_hit,
    "miss.wav": lambda: chirp(0.13, 350, 110, 13.0),
    "ready.wav": lambda: arpeggio([523, 784, 1047, 1568], step=0.055, peak=0.48),
    "start.wav": lambda: arpeggio([392, 523, 659, 1047], step=0.07),
    "stop.wav": lambda: arpeggio([659, 392], step=0.08, peak=0.45),
    "select.wav": lambda: arpeggio([784, 1047], step=0.04, tail=0.05, peak=0.34),
    "victory.wav": lambda: arpeggio([523, 659, 784, 1047, 1319], step=0.11),
    "gameover.wav": lambda: arpeggio([523, 466, 392, 262], step=0.15, tail=0.2),
}


def write_wav(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(signal, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    made = []
    for name, fn in SOUNDS.items():
        path = OUT / name
        if args.ensure and path.exists() and path.stat().st_size > 44:
            continue
        write_wav(path, fn())
        made.append(name)
    if made:
        print("Generated:", ", ".join(made))


if __name__ == "__main__":
    main()
