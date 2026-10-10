from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class Config:
    # Browser capture / analysis. Camera itself is never opened by Python.
    camera_width: int = 640
    camera_height: int = 480
    analysis_width: int = 416
    analysis_fps: int = 15
    jpeg_quality: float = 0.68

    # Vision controls.
    hand_gain: float = 1.25
    hand_smoothing: float = 0.42
    smile_threshold: float = 1.55
    blink_threshold: float = 0.15  # legacy/fallback tuning retained for config compatibility
    blink_release: float = 0.19
    blink_close_ratio: float = 0.58
    blink_release_ratio: float = 0.78
    blink_cooldown: float = 0.22
    fist_max_extended: int = 1
    fist_release_extended: int = 3

    # Deliberate wide-mouth gesture -> independent triple shot.  Hysteresis and
    # a short hold avoid accidental triggering while merely speaking.
    mouth_open_threshold: float = 0.38
    mouth_release_threshold: float = 0.22
    mouth_hold_frames: int = 2
    mouth_cooldown: float = 0.65

    # Charge / super.
    blink_charge_step: float = 0.25
    voice_charge_step: float = 0.25
    super_decay_delay: float = 4.0
    super_decay_rate: float = 0.0

    # Game.
    level_duration: int = 75
    level_difficulty: int = 1
    target_radius: int = 34
    best_score: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def load(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self.update(**data)

    def save(self) -> None:
        try:
            CONFIG_PATH.write_text(
                json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def update(self, **patch) -> dict:
        known = {f.name: type(f.default) for f in fields(self)}
        for key, raw in patch.items():
            typ = known.get(key)
            if typ is None or isinstance(raw, bool):
                continue
            try:
                value = typ(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if key == "analysis_fps" and not 5 <= value <= 30:
                continue
            if key == "analysis_width" and not 240 <= value <= 640:
                continue
            if key == "jpeg_quality" and not 0.35 <= value <= 0.95:
                continue
            if key == "hand_gain" and not 0.5 <= value <= 2.5:
                continue
            if key == "hand_smoothing" and not 0.05 <= value <= 1.0:
                continue
            if key == "smile_threshold" and not 1.1 <= value <= 2.4:
                continue
            if key == "blink_threshold" and not 0.08 <= value <= 0.35:
                continue
            if key == "blink_release" and not 0.10 <= value <= 0.45:
                continue
            if key == "blink_close_ratio" and not 0.35 <= value <= 0.80:
                continue
            if key == "blink_release_ratio" and not 0.55 <= value <= 0.98:
                continue
            if key == "blink_cooldown" and not 0.10 <= value <= 1.0:
                continue
            if key == "mouth_open_threshold" and not 0.18 <= value <= 0.80:
                continue
            if key == "mouth_release_threshold" and not 0.08 <= value <= 0.60:
                continue
            if key == "mouth_hold_frames" and not 1 <= value <= 8:
                continue
            if key == "mouth_cooldown" and not 0.20 <= value <= 2.0:
                continue
            if key in {"blink_charge_step", "voice_charge_step"} and not 0.05 <= value <= 1.0:
                continue
            if key == "super_decay_delay" and not 0.5 <= value <= 15.0:
                continue
            if key == "super_decay_rate" and not 0.0 <= value <= 0.5:
                continue
            if key == "level_duration" and not 15 <= value <= 600:
                continue
            if key == "level_difficulty" and not 1 <= value <= 5:
                continue
            if key == "target_radius" and not 18 <= value <= 70:
                continue
            setattr(self, key, value)
        self.save()
        return self.as_dict()


CFG = Config()
CFG.load()
