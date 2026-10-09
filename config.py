# smilegun/config.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, fields
from pathlib import Path

# Настройки и рекорд хранятся здесь и переживают перезапуск сервера.
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


@dataclass
class Config:
    # Поле
    field_width: int = 900
    field_height: int = 950

    # Старые параметры усиления взгляда сохранены для совместимости.
    # Прицел управляется рукой; взгляд отображается только в диагностике.
    iris_gain: float = 2.6
    iris_gain_y: float = 1.6
    smoothing_alpha: float = 0.30   # EMA (0..1, меньше = плавнее)

    # Нейтраль взгляда: кнопка «Калибровка» запоминает диагностический центр.
    gaze_neutral_x: float = 0.0
    gaze_neutral_y: float = 0.0

    # Старый параметр мёртвой зоны взгляда, сохранён для совместимости.
    gaze_deadzone: float = 0.02

    # Старые пороги лица; улыбка и раскрытие рта доступны как диагностика.
    smile_threshold: float = 1.55
    smile_hysteresis: float = 0.65  # отпускание: ниже threshold * 0.65

    # mouth_open_score — просвет между внутренними губами / ширина глаза.
    mouth_open_threshold: float = 0.30

    # Игра
    bullet_speed: float = 9.0       # пикс/кадр
    level_duration: int = 60       # секунд на уровень, 10..600
    level_difficulty: int = 1      # сложность: 1..5

    # Камера
    camera_index: int = 0

    # Управление рукой (MediaPipe Hands): рука двигает прицел и стреляет кулаком.
    hand_gain: float = 1.2          # усиление движения ладони вокруг центра
    fist_threshold: float = 1.0     # раскрытие ладони ниже этого = кулак
    fist_release: float = 1.25      # выше этого кулак «отпускается» (гистерезис)

    # Супер-сила: энергия моргания складывается, после паузы заряд затухает.
    super_blink_gain: float = 0.25     # доля полного заряда на одно моргание
    super_decay_delay: float = 3.0     # пауза перед затуханием, секунд
    super_decay_rate: float = 0.08     # сколько заряда теряется за секунду
    # Сохранены для совместимости старых настроек; заряд от движения глаз отключён.
    super_blink_count: int = 4
    super_charge_rate: float = 0.6
    super_charge_move_min: float = 0.03
    super_blink_window: float = 2.0

    # Рекорд (лучший счёт), сохраняется в config.json.
    best_score: int = 0

    def update(self, **kwargs):
        types = {item.name: type(item.default) for item in fields(self)}
        for k, v in kwargs.items():
            if k not in types or isinstance(v, bool):
                continue
            try:
                converted = types[k](v)
                if not math.isfinite(converted):
                    continue
                if types[k] is int and float(v) != converted:
                    continue
                if k == "level_duration" and not 10 <= converted <= 600:
                    continue
                if k == "level_difficulty" and not 1 <= converted <= 5:
                    continue
                if k == "super_blink_count" and not 2 <= converted <= 8:
                    continue
                if k == "super_blink_gain" and not 0.05 <= converted <= 1.0:
                    continue
                if k == "super_decay_delay" and not 0.5 <= converted <= 15.0:
                    continue
                if k == "super_decay_rate" and not 0.0 <= converted <= 0.5:
                    continue
                setattr(self, k, converted)
            except (TypeError, ValueError, OverflowError):
                # Невалидное значение (строка вместо числа и т.п.) пропускаем,
                # чтобы один кривой запрос не ронял /config.
                continue
        return asdict(self)

    def as_dict(self):
        return asdict(self)

    def save(self) -> None:
        """Записать конфиг на диск. Ошибки не фатальны: игра работает и без файла."""
        try:
            CONFIG_FILE.write_text(
                json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    def load(self) -> None:
        """Поднять сохранённый конфиг при старте (если файл есть и валиден)."""
        if not CONFIG_FILE.exists():
            return
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            # При первом запуске с новым зарядом сохраняем привычную силу моргания.
            if "super_blink_gain" not in data and "super_blink_count" in data:
                try:
                    count = int(data["super_blink_count"])
                    if not isinstance(data["super_blink_count"], bool) and 2 <= count <= 8 and float(data["super_blink_count"]) == count:
                        data["super_blink_gain"] = 1.0 / count
                except (TypeError, ValueError, OverflowError):
                    pass
            self.update(**data)


CFG = Config()
CFG.load()
