from __future__ import annotations

import argparse
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
    if not np.isfinite(signal).all():
        raise ValueError(f'Invalid audio samples: {out.name}')
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


def finish(signal: np.ndarray, peak=0.78) -> np.ndarray:
    """Мягкая атака и окончание без щелчков, запас громкости для наложения."""
    signal = signal.copy()
    attack = min(int(SR * 0.006), len(signal))
    release = min(int(SR * 0.035), len(signal))
    signal[:attack] *= np.linspace(0, 1, attack)
    signal[-release:] *= np.linspace(1, 0, release)
    maximum = np.max(np.abs(signal))
    if maximum:
        signal *= peak / maximum
    return signal.astype(np.float32)


def charge_pulse(duration=0.38):
    """Короткий нарастающий импульс энергии на каждое моргание."""
    n = int(SR * duration)
    t = np.arange(n) / SR
    frequency = 180 * np.power(5.5, t / duration)
    phase = 2 * np.pi * np.cumsum(frequency) / SR
    pulse = np.sin(phase) + 0.28 * np.sin(phase * 2.01) + 0.12 * np.sin(phase * 3)
    envelope = np.sin(np.pi * t / duration) ** 0.8
    shimmer = np.sin(2 * np.pi * 1568 * t) * np.exp(-np.maximum(0, t - 0.15) * 10)
    shimmer *= np.clip((t - 0.15) * 25, 0, 1)
    return finish(lowpass(pulse * envelope + shimmer * 0.16, 8500), 0.68)


def arpeggio(notes, step=0.085, tail=0.12, peak=0.65):
    """Чистые короткие ноты: отдельная мелодия для каждого действия."""
    signal = np.zeros(int(SR * (len(notes) * step + tail)))
    for index, frequency in enumerate(notes):
        offset = int(index * step * SR)
        t = np.arange(int((step + tail) * SR)) / SR
        voice = np.sin(2 * np.pi * frequency * t)
        voice += 0.18 * np.sin(4 * np.pi * frequency * t)
        voice *= np.exp(-t / (step * 0.75))
        voice[:int(SR * 0.003)] *= np.linspace(0, 1, int(SR * 0.003))
        length = min(len(voice), len(signal) - offset)
        signal[offset:offset + length] += voice[:length]
    return finish(signal, peak)


def impact(duration=0.19):
    t = np.arange(int(SR * duration)) / SR
    noise = lowpass(np.random.default_rng(37).normal(0, 1, len(t)), 7000)
    strike = np.sin(2 * np.pi * 740 * t) + 0.35 * np.sin(2 * np.pi * 1480 * t)
    return finish((strike + noise * 0.24) * np.exp(-t * 29), 0.73)


def miss(duration=0.15):
    t = np.arange(int(SR * duration)) / SR
    phase = 2 * np.pi * np.cumsum(np.linspace(360, 110, len(t))) / SR
    return finish(np.sin(phase) * np.exp(-t * 17), 0.35)


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


SOUNDS = {
    'blaster.mp3': blaster,
    'charge.mp3': charge_pulse,
    'super.mp3': super_blast,
    'select.mp3': lambda: arpeggio([784, 1047], step=0.045, tail=0.07, peak=0.42),
    'start.mp3': lambda: arpeggio([392, 523, 659, 1047], step=0.075),
    'stop.mp3': lambda: arpeggio([659, 392], step=0.075, peak=0.48),
    'countdown.mp3': lambda: arpeggio([880], step=0.06, tail=0.06, peak=0.48),
    'hit.mp3': impact,
    'miss.mp3': miss,
    'level_complete.mp3': lambda: arpeggio([523, 659, 784, 1047], step=0.105),
    'victory.mp3': lambda: arpeggio([523, 659, 784, 1047, 784, 1047, 1319], step=0.125),
    'gameover.mp3': lambda: arpeggio([523, 494, 392, 262], step=0.16, tail=0.24),
    'calibrate.mp3': lambda: arpeggio([587, 740, 587], step=0.08, peak=0.48),
    'camera_on.mp3': lambda: arpeggio([440, 880], step=0.08, peak=0.42),
    'camera_off.mp3': lambda: arpeggio([880, 440], step=0.08, peak=0.42),
}


def main():
    parser = argparse.ArgumentParser(description='Generate the SmileGun arcade sound set locally.')
    parser.add_argument('--ensure', action='store_true', help='Generate only missing assets.')
    args = parser.parse_args()
    generated = []
    for name, synthesize in SOUNDS.items():
        out = STATIC / name
        if args.ensure and out.is_file() and out.stat().st_size:
            continue
        write_mp3(synthesize(), out)
        generated.append(name)
    if generated:
        print(f'Generated {len(generated)} sounds: {", ".join(generated)} -> {STATIC}')


if __name__ == '__main__':
    main()
