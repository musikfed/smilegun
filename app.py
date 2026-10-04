from config import CFG

FIELD_CENTER = 0.5

def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

def hybrid_aim(head_x: float, head_y: float, gaze_x: float, gaze_y: float,
               neutral_x: float, neutral_y: float) -> tuple[float, float]:
    """Гибридный прицел: смещение головы от нейтрали + взгляд.

    Считаем смещение от центра поля, а не абсолютную координату в кадре.
    Раньше было raw_ax = head_x * head_gain + ..., то есть уже ~0.5 при
    взгляде прямо: вместе с добавкой от зрачков значение сразу уходило за
    1.0, и прицел залипал на краю поля.

    Взгляд и голова складываются как два слагаемых одного смещения,
    поэтому центр поля (0.5, 0.5) соответствует нейтральной позе.
    """
    offset_x = gaze_x * CFG.iris_gain
    offset_y = gaze_y * CFG.iris_gain * CFG.iris_gain_y
    return clamp01(FIELD_CENTER + offset_x), clamp01(FIELD_CENTER + offset_y)
