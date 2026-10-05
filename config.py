# smilegun/config.py
from dataclasses import dataclass, asdict

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

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))
        return asdict(self)

    def as_dict(self):
        return asdict(self)

CFG = Config()
