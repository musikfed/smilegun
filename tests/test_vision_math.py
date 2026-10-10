from types import SimpleNamespace

from vision_math import clamp01, count_extended_fingers, eye_openness, mouth_openness, normalized_aim


def p(x, y):
    return SimpleNamespace(x=x, y=y)


def test_clamp_and_aim():
    assert clamp01(-1) == 0
    assert clamp01(2) == 1
    x, y = normalized_aim(0.75, 0.25, 2.0)
    assert x == 1.0
    assert y == 0.0


def test_eye_openness_scale_independent():
    lm = [p(0, 0) for _ in range(400)]
    lm[0], lm[1], lm[2], lm[3] = p(0, 0), p(1, 0), p(0.5, -0.1), p(0.5, 0.1)
    assert abs(eye_openness(lm, 0, 1, 2, 3) - 0.2) < 1e-9


def test_fist_has_few_extended_fingers():
    lm = [p(0, 0) for _ in range(21)]
    # Wrist and index MCP.
    lm[0] = p(0, 0)
    lm[5] = p(0.1, 0.1)
    # Put PIP farther than tips -> folded fingers.
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        lm[pip] = p(0.4, 0)
        lm[tip] = p(0.2, 0)
    lm[3] = p(0.2, 0.2)
    lm[4] = p(0.15, 0.15)
    assert count_extended_fingers(lm) <= 1


def test_mouth_openness_scale_independent():
    lm = [p(0, 0) for _ in range(400)]
    lm[61], lm[291] = p(0, 0), p(1, 0)
    lm[13], lm[14] = p(0.5, -0.2), p(0.5, 0.2)
    assert abs(mouth_openness(lm) - 0.4) < 1e-9

