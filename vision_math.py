from __future__ import annotations

import math
from typing import Sequence


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance(a, b) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def normalized_aim(x: float, y: float, gain: float) -> tuple[float, float]:
    """Amplify hand motion around center while keeping coordinates in [0, 1]."""
    return (
        clamp01(0.5 + (float(x) - 0.5) * float(gain)),
        clamp01(0.5 + (float(y) - 0.5) * float(gain)),
    )


def eye_openness(landmarks: Sequence, outer: int, inner: int, top: int, bottom: int) -> float:
    width = distance(landmarks[outer], landmarks[inner])
    if width < 1e-6:
        return 0.0
    return distance(landmarks[top], landmarks[bottom]) / width


def smile_ratio(landmarks: Sequence) -> float:
    # Mouth width normalized by mean eye width. This is scale-independent and cheap.
    mouth = distance(landmarks[61], landmarks[291])
    left_eye = distance(landmarks[362], landmarks[263])
    right_eye = distance(landmarks[33], landmarks[133])
    eye = (left_eye + right_eye) * 0.5
    return mouth / eye if eye > 1e-6 else 0.0


def mouth_openness(landmarks: Sequence) -> float:
    """Inner-lip gap normalized by mouth width.

    A wide-open mouth produces a much larger value than ordinary speech, which
    lets the game use a deliberate mouth-open gesture without tying it to voice
    recognition.
    """
    width = distance(landmarks[61], landmarks[291])
    if width < 1e-6:
        return 0.0
    return distance(landmarks[13], landmarks[14]) / width


def count_extended_fingers(landmarks: Sequence) -> int:
    """Approximate open fingers using wrist distance; sufficient for fist edge detection."""
    wrist = landmarks[0]
    pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    count = 0
    for tip, pip in pairs:
        if distance(landmarks[tip], wrist) > distance(landmarks[pip], wrist) * 1.10:
            count += 1
    # Thumb: tip should be notably farther from index MCP than its IP joint.
    index_mcp = landmarks[5]
    if distance(landmarks[4], index_mcp) > distance(landmarks[3], index_mcp) * 1.15:
        count += 1
    return count
