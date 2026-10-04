# smilegun/config.py
from dataclasses import dataclass, asdict

@dataclass
class Config:
    # Поле
    field_width: int = 900
    field_height: int = 950

    # Управление взглядом
    # head_gain масштабирует СМЕЩЕНИЕ головы от нейтрали (а не абсолютную
    # позицию в кадре): нейтраль = центр поля. Комфортный поворот головы
    # смещает центр лица примерно на ±0.06 кадра -> ±0.09 поля.
    head_gain: float = 1.5
    head_gain_y: float = 1.0        # отдельный множитель по вертикали (кивок)

    # iris_gain масштабирует взгляд. Взгляд измеряется в ширинах глаза:
    # спокойный перевод взгляда в угол экрана — это примерно ±0.3 ширины
    # глаза, то есть при iris_gain 2.6 прицел проходит ±0.8 поля.
    # Много не ставь: зрачок ходит мало, и большой множитель сразу упирает
    # прицел в край поля.
    iris_gain: float = 2.6
    iris_gain_y: float = 1.6        # взгляд по вертикали ходит меньше

    eye_deadzone: float = 0.03          # мёртвая зона взгляда (доля ширины глаза)
    head_recenter_rate: float = 0.006   # авто-подстройка нейтрали головы (0 = выкл.)
    head_recenter_tol: float = 0.004    # насколько спокойным должен быть взгляд
    smoothing_alpha: float = 0.30   # EMA (0..1, меньше = плавнее)

    # Улыбка
    smile_threshold: float = 0.10   # порог smile_score
    smile_hysteresis: float = 0.65  # отпускание порога (0.65 = 65% от threshold)

    # Игра
    fire_cooldown: float = 0.20     # сек между выстрелами
    bullet_speed: float = 9.0       # пикс/кадр

    # Камера
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))
        return asdict(self)

    def as_dict(self):
        return asdict(self)

CFG = Config()
