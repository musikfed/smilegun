from __future__ import annotations

import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from config import CFG
from vision_math import (
    clamp01,
    count_extended_fingers,
    eye_openness,
    mouth_openness,
    normalized_aim,
    smile_ratio,
)


class VisionEngine:
    """Analyze browser-provided frames and emit game-control state.

    Python never opens the webcam. The browser owns getUserMedia(); this module
    receives JPEG frames and runs the light MediaPipe control logic.

    Firing sources intentionally use independent counters. A fist, a spoken
    "ПИУ", a keyboard shot and a mouth-open triple shot can therefore never
    block or unlock one another by accident.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )
        self._aim_x = 0.5
        self._aim_y = 0.5
        self._fist_held = False
        self._prev_smile = False

        # Blink detector is adaptive. Webcam/face geometry varies too much for a
        # single hard-coded EAR threshold to be reliable for everyone.
        self._eye_baseline: float | None = None
        self._blink_latched = False
        self._last_blink_event = 0.0
        self._face_missing_since: float | None = None

        # A deliberate wide mouth is a separate gesture from speech. Require a
        # short hold + hysteresis so normal talking usually stays below it.
        self._mouth_latched = False
        self._mouth_open_frames = 0
        self._last_mouth_event = 0.0

        self._charge = 0.0
        self._charge_updated_at = time.monotonic()
        self._last_charge_event = self._charge_updated_at

        # Compatibility/global shot counter plus independent source counters.
        self._shot_id = 0
        self._shot_power = 0.0
        self._hand_shot_id = 0
        self._voice_shot_id = 0
        self._manual_shot_id = 0
        self._mouth_shot_id = 0
        self._hand_shot_power = 0.0
        self._voice_shot_power = 0.0
        self._manual_shot_power = 0.0
        self._mouth_shot_power = 0.0

        self._charge_id = 0
        self._blink_id = 0
        self._super_id = 0
        self._frames = 0
        self._fps_window_start = time.monotonic()
        self._fps = 0.0
        self._last_state = self._empty_state()

    def _shot_fields(self) -> dict:
        return {
            "shot_id": self._shot_id,
            "shot_power": self._shot_power,
            "hand_shot_id": self._hand_shot_id,
            "hand_shot_power": self._hand_shot_power,
            "voice_shot_id": self._voice_shot_id,
            "voice_shot_power": self._voice_shot_power,
            "manual_shot_id": self._manual_shot_id,
            "manual_shot_power": self._manual_shot_power,
            "mouth_shot_id": self._mouth_shot_id,
            "mouth_shot_power": self._mouth_shot_power,
        }

    def _empty_state(self) -> dict:
        state = {
            "source": "browser",
            "camera": False,
            "face": False,
            "hand_present": False,
            "aim_x": 0.5,
            "aim_y": 0.5,
            "fist": False,
            "extended_fingers": 0,
            "blink": False,
            "blink_score": 0.0,
            "blink_threshold": 0.0,
            "blink_release": 0.0,
            "eye_baseline": 0.0,
            "blink_id": self._blink_id,
            "mouth_open": False,
            "mouth_score": 0.0,
            "mouth_threshold": float(CFG.mouth_open_threshold),
            "mouth_release": float(CFG.mouth_release_threshold),
            "charge_id": self._charge_id,
            "smile": False,
            "smile_score": 0.0,
            "super_charge": self._charge,
            "super_ready": self._charge >= 0.999,
            "super_id": self._super_id,
            "fps": self._fps,
            "frames_processed": self._frames,
            "processing_ms": 0.0,
            "error": None,
        }
        state.update(self._shot_fields())
        return state

    def close(self) -> None:
        with self._lock:
            self._face_mesh.close()
            self._hands.close()

    def reset_runtime(self) -> dict:
        with self._lock:
            self._aim_x = self._aim_y = 0.5
            self._fist_held = False
            self._prev_smile = False
            self._eye_baseline = None
            self._blink_latched = False
            self._last_blink_event = 0.0
            self._face_missing_since = None
            self._mouth_latched = False
            self._mouth_open_frames = 0
            self._last_mouth_event = 0.0
            self._charge = 0.0
            self._shot_power = 0.0
            now = time.monotonic()
            self._charge_updated_at = now
            self._last_charge_event = now
            self._last_state = self._empty_state()
            return dict(self._last_state)

    def _add_charge(self, amount: float) -> None:
        before = self._charge
        self._charge = clamp01(self._charge + amount)
        self._charge_updated_at = time.monotonic()
        self._last_charge_event = self._charge_updated_at
        if self._charge > before + 1e-6:
            self._charge_id += 1

    def _trigger_shot(self, source: str) -> None:
        """Consume the current square charge and emit one source-specific event."""
        power = self._charge
        self._shot_power = power
        self._shot_id += 1

        if source == "hand":
            self._hand_shot_id += 1
            self._hand_shot_power = power
        elif source == "voice":
            self._voice_shot_id += 1
            self._voice_shot_power = power
        elif source == "mouth":
            self._mouth_shot_id += 1
            self._mouth_shot_power = power
        else:
            self._manual_shot_id += 1
            self._manual_shot_power = power

        self._charge = 0.0
        now = time.monotonic()
        self._charge_updated_at = now
        self._last_charge_event = now

    def action(self, action: str, source: str = "manual") -> dict:
        with self._lock:
            if action == "fire":
                # Browser voice and keyboard/manual input remain independent from
                # the hand and mouth detectors running in process_bgr().
                self._trigger_shot("voice" if source == "voice" else "manual")
            elif action == "charge":
                self._add_charge(CFG.voice_charge_step)
            elif action == "reset_charge":
                self._charge = 0.0

            state = dict(self._last_state)
            state.update(self._shot_fields())
            state.update(
                charge_id=self._charge_id,
                super_charge=self._charge,
                super_ready=self._charge >= 0.999,
            )
            self._last_state = state
            return state

    def state(self) -> dict:
        with self._lock:
            return dict(self._last_state)

    def _decay_charge(self, now: float) -> None:
        if CFG.super_decay_rate <= 0:
            self._charge_updated_at = now
            return
        dt = max(0.0, now - self._charge_updated_at)
        self._charge_updated_at = now
        idle = now - self._last_charge_event
        if self._charge > 0 and idle > CFG.super_decay_delay:
            self._charge = clamp01(self._charge - CFG.super_decay_rate * dt)

    def process_jpeg(self, payload: bytes) -> dict:
        if not payload:
            state = self.state()
            state["error"] = "empty frame"
            return state
        data = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            state = self.state()
            state["error"] = "jpeg decode failed"
            return state
        return self.process_bgr(frame)

    def process_bgr(self, frame: np.ndarray) -> dict:
        started = time.perf_counter()
        with self._lock:
            now = time.monotonic()
            self._decay_charge(now)

            h, w = frame.shape[:2]
            if w > 720:
                scale = 720.0 / w
                frame = cv2.resize(frame, (720, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_result = self._face_mesh.process(rgb)
            hand_result = self._hands.process(rgb)

            face_present = bool(face_result.multi_face_landmarks)
            hand_present = bool(hand_result.multi_hand_landmarks)
            blink = False
            blink_score = 0.0
            blink_threshold = 0.0
            blink_release = 0.0
            mouth_open = False
            mouth_score = 0.0
            smile = False
            smile_score_value = 0.0
            fist = False
            extended = 0

            if face_present:
                self._face_missing_since = None
                lm = face_result.multi_face_landmarks[0].landmark
                left = eye_openness(lm, 362, 263, 386, 374)
                right = eye_openness(lm, 33, 133, 159, 145)
                blink_score = (left + right) * 0.5

                # Learn the user's normal open-eye value. Only update the baseline
                # while the eyes look plausibly open, so a blink cannot drag it down.
                if self._eye_baseline is None:
                    if blink_score > 0.06:
                        self._eye_baseline = blink_score
                else:
                    if not self._blink_latched and blink_score > self._eye_baseline * 0.70:
                        alpha = 0.045
                        self._eye_baseline = (1.0 - alpha) * self._eye_baseline + alpha * blink_score
                    elif blink_score > self._eye_baseline:
                        self._eye_baseline = self._eye_baseline * 0.97 + blink_score * 0.03

                baseline = self._eye_baseline or max(blink_score, 0.20)
                blink_threshold = max(0.065, min(0.22, baseline * CFG.blink_close_ratio))
                blink_release = max(blink_threshold + 0.018, min(0.30, baseline * CFG.blink_release_ratio))
                blink = blink_score <= blink_threshold

                # Charge on the CLOSING edge. This is more robust than waiting for
                # reopen because some webcams briefly lose face tracking at full blink.
                if blink and not self._blink_latched and now - self._last_blink_event >= CFG.blink_cooldown:
                    self._blink_latched = True
                    self._last_blink_event = now
                    self._blink_id += 1
                    self._add_charge(CFG.blink_charge_step)
                elif self._blink_latched and blink_score >= blink_release:
                    self._blink_latched = False

                # Wide-mouth gesture -> independent triple shot. It is intentionally
                # stricter than normal speech so saying "ПИУ" should normally remain
                # the red voice shot, while an exaggerated open mouth is blue triple.
                mouth_score = mouth_openness(lm)
                mouth_open = mouth_score >= CFG.mouth_open_threshold
                if mouth_open:
                    self._mouth_open_frames += 1
                    if (
                        not self._mouth_latched
                        and self._mouth_open_frames >= CFG.mouth_hold_frames
                        and now - self._last_mouth_event >= CFG.mouth_cooldown
                    ):
                        self._mouth_latched = True
                        self._last_mouth_event = now
                        self._trigger_shot("mouth")
                elif mouth_score <= CFG.mouth_release_threshold:
                    self._mouth_open_frames = 0
                    self._mouth_latched = False

                smile_score_value = smile_ratio(lm)
                smile = smile_score_value >= CFG.smile_threshold
                if smile and not self._prev_smile and self._charge >= 0.999:
                    self._super_id += 1
                    self._charge = 0.0
                    self._last_charge_event = now
                self._prev_smile = smile
            else:
                self._prev_smile = False
                self._mouth_open_frames = 0
                if self._face_missing_since is None:
                    self._face_missing_since = now
                # Do not instantly clear latches on one dropped frame.
                if now - self._face_missing_since > 0.55:
                    self._blink_latched = False
                    self._mouth_latched = False

            if hand_present:
                lm = hand_result.multi_hand_landmarks[0].landmark
                palm_ids = (0, 5, 9, 13, 17)
                px = sum(lm[i].x for i in palm_ids) / len(palm_ids)
                py = sum(lm[i].y for i in palm_ids) / len(palm_ids)
                raw_x, raw_y = normalized_aim(px, py, CFG.hand_gain)
                a = CFG.hand_smoothing
                self._aim_x = a * raw_x + (1 - a) * self._aim_x
                self._aim_y = a * raw_y + (1 - a) * self._aim_y

                extended = count_extended_fingers(lm)
                fist = extended <= CFG.fist_max_extended
                if fist and not self._fist_held:
                    self._trigger_shot("hand")
                    self._fist_held = True
                elif extended >= CFG.fist_release_extended:
                    self._fist_held = False
            else:
                self._fist_held = False

            self._frames += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self._fps = self._frames / elapsed
                self._frames = 0
                self._fps_window_start = now

            processing_ms = (time.perf_counter() - started) * 1000.0
            state = {
                "source": "browser",
                "camera": True,
                "face": face_present,
                "hand_present": hand_present,
                "aim_x": clamp01(self._aim_x),
                "aim_y": clamp01(self._aim_y),
                "fist": fist,
                "extended_fingers": extended,
                "blink": blink,
                "blink_score": round(blink_score, 4),
                "blink_threshold": round(blink_threshold, 4),
                "blink_release": round(blink_release, 4),
                "eye_baseline": round(self._eye_baseline or 0.0, 4),
                "blink_id": self._blink_id,
                "mouth_open": mouth_open,
                "mouth_score": round(mouth_score, 4),
                "mouth_threshold": round(float(CFG.mouth_open_threshold), 4),
                "mouth_release": round(float(CFG.mouth_release_threshold), 4),
                "charge_id": self._charge_id,
                "smile": smile,
                "smile_score": round(smile_score_value, 4),
                "super_charge": round(self._charge, 4),
                "super_ready": self._charge >= 0.999,
                "super_id": self._super_id,
                "fps": round(self._fps, 1),
                "processing_ms": round(processing_ms, 1),
                "error": None,
            }
            state.update(self._shot_fields())
            self._last_state = state
            return dict(state)
