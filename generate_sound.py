from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'static' / 'blaster.mp3'


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, x)


def blaster(duration=0.42, f_start=2450.0, f_end=95.0):
    n = int(SR * duration)
    t = np.arange(n) / SR
    # Exponential frequency fall: sharp sci-fi "pew".
    k = np.log(f_start / f_end) / duration
    freq = f_start * np.exp(-k * t)
    phase = 2 * np.pi * np.cumsum(freq) / SR

    core = 0.72 * np.sin(phase)
    core += 0.20 * np.sin(phase * 1.018 + 0.35)
    core += 0.08 * np.sin(phase * 0.51)

    # Short click/impact at the muzzle.
    click = np.random.default_rng(7).normal(0, 1, n)
    click = lowpass(click, 8500)
    click *= np.exp(-t * 45)

    # Metallic tail.
    grit = np.random.default_rng(9).normal(0, 1, n)
    grit = lowpass(grit, 4200)
    grit *= np.exp(-t * 14)

    env = np.exp(-t * 8.8)
    attack = min(int(SR * 0.0025), n)
    env[:attack] *= np.linspace(0, 1, attack)

    signal = core * env + 0.18 * click + 0.10 * grit
    signal = np.tanh(signal * 1.8)
    signal = lowpass(signal, 11500)
    peak = np.max(np.abs(signal))
    if peak:
        signal = signal / peak * 0.88
    return signal.astype(np.float32)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    signal = blaster()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / 'blaster.wav'
        wavfile.write(wav, SR, (signal * 32767).astype(np.int16))
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(wav), '-codec:a', 'libmp3lame', '-b:a', '192k', str(OUT)
        ], check=True)
    print(f'Generated: {OUT}')


if __name__ == '__main__':
    main()
