# smilegun/config.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Настройки и рекорд хранятся здесь и переживают перезапуск сервера.
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


@dataclass
class Config:
    # Поле
    field_width: int = 900
    field_height: int = 950

    # Управление взглядом.
    # Кубик двигают ТОЛЬКО зрачки: взгляд измеряется как
    # (зрачок - центр своего глаза) / ширина глаза, поэтому движение головы
    # в кадре взаимно сокращается и в прицел не попадает.
    # Спокойный перевод взгляда в угол экрана — примерно ±0.25 ширины глаза,
    # то есть при iris_gain 2.6 прицел проходит ±0.65 поля. Много не ставь:
    # зрачок ходит мало, и большой множитель сразу упирает прицел в край поля.
    iris_gain: float = 2.6
    iris_gain_y: float = 1.6        # взгляд по вертикали ходит меньше
    smoothing_alpha: float = 0.30   # EMA (0..1, меньше = плавнее)

    # Нейтраль взгляда: заполняется кнопкой «Калибровка» (смотреть в центр).
    # Прицел считается смещением от неё, поэтому посадка перед камерой и
    # размер лица не важны — важно только движение зрачков.
    gaze_neutral_x: float = 0.0
    gaze_neutral_y: float = 0.0

    # Мёртвая зона взгляда вокруг нейтрали (в долях ширины глаза): гасит
    # микро-дрожание зрачка, из-за которого прицел трясётся на месте.
    gaze_deadzone: float = 0.02

    # Улыбка = рот, растянутый в ширину (губы можно сжать, рот открывать не
    # надо). smile_score — ширина рта / ширина глаза: в покое ~1.5,
    # при широкой улыбке ~1.8.
    smile_threshold: float = 1.55
    smile_hysteresis: float = 0.65  # отпускание: ниже threshold * 0.65

    # Широко открытый рот = двойной выстрел.
    # mouth_open_score — просвет между внутренними губами / ширина глаза.
    mouth_open_threshold: float = 0.30

    # Игра
    bullet_speed: float = 9.0       # пикс/кадр

    # Камера
    camera_index: int = 0

    # Управление рукой (MediaPipe Hands): рука двигает прицел и стреляет кулаком.
    hand_gain: float = 1.2          # усиление движения ладони вокруг центра
    fist_threshold: float = 1.0     # раскрытие ладони ниже этого = кулак
    fist_release: float = 1.25      # выше этого кулак «отпускается» (гистерезис)

    # Супер-сила: движение глаз заряжает, N морганий подряд = супер-выстрел.
    super_charge_rate: float = 0.6      # скорость зарядки от величины движения глаз
    super_charge_move_min: float = 0.03 # движение глаз меньше этого заряд не растит
    super_blink_count: int = 4          # сколько морганий нужно
    super_blink_window: float = 2.0     # окно (сек), за которое надо успеть моргнуть

    # Рекорд (лучший счёт), сохраняется в config.json.
    best_score: int = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                continue
            try:
                setattr(self, k, type(getattr(self, k))(v))
            except (TypeError, ValueError):
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
            self.update(**data)


CFG = Config()
CFG.load()
