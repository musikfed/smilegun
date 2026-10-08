from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
ROOT = Path(__file__).resolve().parent
STATIC = ROOT / 'static'


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, x)


def write_mp3(signal: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / 'tmp.wav'
        wavfile.write(wav, SR, (signal * 32767).astype(np.int16))
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(wav), '-codec:a', 'libmp3lame', '-b:a', '192k', str(out)
        ], check=True)


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


def charge_loop(duration=1.6):
    """Бесшовная петля «зарядки»: гул из кратных гармоник + тремоло.

    Частоты подобраны так, чтобы за duration укладывалось целое число периодов,
    поэтому mp3 можно зациклить без щелчка на стыке.
    """
    n = int(SR * duration)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    base = 100.0
    for k, amp in ((1, 0.5), (2, 0.28), (3, 0.15), (5, 0.07), (7, 0.04)):
        sig += amp * np.sin(2 * np.pi * base * k * t)
    # Тремоло: целое число периодов LFO на длительность — бесшовно.
    lfo = 1.0 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t / duration))
    sig *= lfo
    peak = np.max(np.abs(sig))
    if peak:
        sig = sig / peak * 0.5
    return sig.astype(np.float32)


def super_blast(duration=0.9):
    """Супер-выстрел: падающий свип + низкий удар + шумовой всплеск."""
    n = int(SR * duration)
    t = np.arange(n) / SR
    f_start, f_end = 1400.0, 55.0
    k = np.log(f_start / f_end) / duration
    freq = f_start * np.exp(-k * t)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    core = 0.6 * np.sin(phase) + 0.3 * np.sin(phase * 0.5) + 0.15 * np.sin(phase * 2.02)
    sub = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 6)
    noise = lowpass(np.random.default_rng(11).normal(0, 1, n), 4000)
    noise *= np.exp(-t * 18)
    env = np.exp(-t * 5)
    sig = core * env + 0.55 * sub + 0.42 * noise
    sig = np.tanh(sig * 1.6)
    sig = lowpass(sig, 12000)
    peak = np.max(np.abs(sig))
    if peak:
        sig = sig / peak * 0.95
    return sig.astype(np.float32)


def main():
    write_mp3(blaster(), STATIC / 'blaster.mp3')
    write_mp3(charge_loop(), STATIC / 'charge.mp3')
    write_mp3(super_blast(), STATIC / 'super.mp3')
    print(f'Generated: blaster.mp3, charge.mp3, super.mp3 -> {STATIC}')


if __name__ == '__main__':
    main()
