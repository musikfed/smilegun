"""Camera-free regression tests for the SmileGun API and input gestures.

Run with: .\\.venv\\Scripts\\python.exe -m unittest -v test_final
All settings writes go to a temporary directory, never the player's config.json.
"""

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Mock constructors before importing the application to avoid tracking sessions.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
import mediapipe as mp
import config as settings

with patch.object(mp.solutions.face_mesh, "FaceMesh"), \
        patch.object(mp.solutions.hands, "Hands"):
    game = importlib.import_module("app")


def hand_landmarks(x=0.5, y=0.5, openness=1.5):
    """Synthetic palm with independent position and finger extension."""
    points = [SimpleNamespace(x=x, y=y) for _ in range(21)]
    points[0].y = y + 0.05
    points[9].y = y - 0.05
    for index in game.CameraTracker.HAND_TIPS:
        points[index].x = x + openness * 0.100001
    return SimpleNamespace(landmark=points)


def face_landmarks(gaze_x=0.0, gaze_y=0.0, closed=False):
    """Synthetic eyes expose blink and gaze inputs without a video device."""
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    height = 0.004 if closed else 0.024
    for outer, inner, top, bottom, iris, center_x in (
        (game.LEFT_EYE_OUTER, game.LEFT_EYE_INNER,
         game.LEFT_EYE_TOP, game.LEFT_EYE_BOTTOM,
         game.LEFT_IRIS_CENTER, 0.4),
        (game.RIGHT_EYE_OUTER, game.RIGHT_EYE_INNER,
         game.RIGHT_EYE_TOP, game.RIGHT_EYE_BOTTOM,
         game.RIGHT_IRIS_CENTER, 0.6),
    ):
        points[outer] = SimpleNamespace(x=center_x - 0.05, y=0.5)
        points[inner] = SimpleNamespace(x=center_x + 0.05, y=0.5)
        points[top] = SimpleNamespace(x=center_x, y=0.5 - height / 2)
        points[bottom] = SimpleNamespace(x=center_x, y=0.5 + height / 2)
        points[iris] = SimpleNamespace(
            x=center_x + gaze_x * 0.1,
            y=0.5 + gaze_y * max(height, game.MIN_EYE_WIDTH),
        )
    points[game.MOUTH_LEFT].x = 0.44
    points[game.MOUTH_RIGHT].x = 0.56
    points[game.LIP_TOP_INNER].y = 0.49
    points[game.LIP_BOTTOM_INNER].y = 0.51
    return SimpleNamespace(landmark=points)


def actual_mediapipe_face(closed=False):
    """Use literal MediaPipe indices, independent of the app's constants.

    Different centers and sizes ensure swapping eyes cannot cancel out when
    averaging two gaze measurements. Index 130 deliberately remains a sentinel.
    """
    points = [SimpleNamespace(x=-1.0, y=-1.0) for _ in range(478)]
    for iris, outer, inner, top, bottom, x, y, width, height in (
        (473, 362, 263, 386, 374, 0.7, 0.6, 0.10, 0.030),
        (468, 133, 33, 159, 145, 0.28, 0.4, 0.16, 0.048),
    ):
        if closed:
            height = width * 0.04
        points[iris] = SimpleNamespace(x=x, y=y)
        points[outer] = SimpleNamespace(x=x - width / 2, y=y)
        points[inner] = SimpleNamespace(x=x + width / 2, y=y)
        points[top] = SimpleNamespace(x=x, y=y - height / 2)
        points[bottom] = SimpleNamespace(x=x, y=y + height / 2)
    points[61] = SimpleNamespace(x=0.42, y=0.7)
    points[291] = SimpleNamespace(x=0.55, y=0.7)
    points[13] = SimpleNamespace(x=0.5, y=0.69)
    points[14] = SimpleNamespace(x=0.5, y=0.71)
    return SimpleNamespace(landmark=points)


class SmileGunTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.config_file = Path(directory) / "config.json"
        self.cfg = settings.Config()
        self.stack.enter_context(patch.object(settings, "CONFIG_FILE", self.config_file))
        self.stack.enter_context(patch.object(settings, "CFG", self.cfg))
        self.stack.enter_context(patch.object(game, "CFG", self.cfg))
        self.stack.enter_context(patch.object(mp.solutions.face_mesh, "FaceMesh"))
        self.stack.enter_context(patch.object(mp.solutions.hands, "Hands"))
        self.clock = 100.0
        self.stack.enter_context(patch.object(game.time, "monotonic", side_effect=lambda: self.clock))
        self.tracker = game.CameraTracker()
        self.stack.enter_context(patch.object(game, "tracker", self.tracker))
        self.stack.enter_context(patch.dict(game.app.config, TESTING=True))
        self.client = game.app.test_client()
        self.tracker._detect_hand = Mock(return_value=None)
        self.tracker._make_mesh = Mock(return_value=None)
        # Keep smoothing separate from tests of gesture semantics.
        for name in ("hand_flt_x", "hand_flt_y", "gaze_flt_x", "gaze_flt_y"):
            setattr(self.tracker, name, Mock(update=lambda value, when: value))

    def process(self, hand=None, face=None, elapsed=0.05):
        self.clock += elapsed
        self.tracker._detect_hand.return_value = hand
        self.tracker._make_mesh.return_value = face
        self.tracker._process_frame(object())
        return self.tracker.get_state()

    def blink(self, elapsed=0.05):
        self.process(face=face_landmarks(closed=True), elapsed=elapsed)
        return self.process(face=face_landmarks(), elapsed=elapsed)


class ApiTests(SmileGunTestCase):
    def test_game_and_read_routes_expose_documented_data(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data().decode("utf-8", errors="strict")
        self.assertIn("smilegun", html[:1000].lower())
        self.assertNotIn("\ufffd", html)
        for label in ("Моргай", "Сложность", "Длительность", "КАЛИБРОВКА"):
            self.assertIn(label, html)

        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        config = response.get_json()
        self.assertTrue({
            "hand_gain", "super_charge_rate", "super_blink_count",
            "super_blink_gain", "super_decay_delay", "super_decay_rate",
            "bullet_speed", "field_width", "field_height", "best_score",
            "gaze_neutral_x", "gaze_neutral_y", "level_duration", "level_difficulty",
        }.issubset(config))

        response = self.client.get("/state")
        self.assertEqual(response.status_code, 200)
        state = response.get_json()
        self.assertTrue({
            "aim_x", "aim_y", "hand_present", "fist", "super_charge",
            "super_ready", "blink_count", "blink_id", "super_id", "charging", "shot_id",
            "frames_read", "face", "fps", "camera_open", "error",
        }.issubset(state))
        self.assertEqual((state["aim_x"], state["aim_y"]), (0.5, 0.5))
        self.assertFalse(state["camera_open"])

    def test_config_update_round_trips_and_ignores_unknown_keys(self):
        response = self.client.post("/config", json={
            "hand_gain": "1.8", "bullet_speed": "invalid",
            "field_width": 1000, "level_duration": 120, "level_difficulty": 3,
            "super_blink_gain": 0.3, "super_decay_delay": 5.5,
            "super_decay_rate": 0.1, "unrecognized_setting": 50,
        })
        self.assertEqual(response.status_code, 200)
        saved = response.get_json()
        self.assertEqual(saved["hand_gain"], 1.8)
        self.assertEqual(saved["field_width"], 1000)
        self.assertEqual(saved["level_duration"], 120)
        self.assertEqual(saved["level_difficulty"], 3)
        self.assertEqual(saved["super_blink_gain"], 0.3)
        self.assertEqual(saved["super_decay_delay"], 5.5)
        self.assertEqual(saved["super_decay_rate"], 0.1)
        self.assertEqual(saved["bullet_speed"], settings.Config().bullet_speed)
        self.assertNotIn("unrecognized_setting", saved)
        restored = settings.Config()
        restored.load()
        self.assertEqual(restored.as_dict(), saved)
        self.assertEqual(self.client.get("/config").get_json(), saved)

    def test_invalid_play_settings_fail_without_partially_saving(self):
        self.cfg.save()
        before = self.config_file.read_bytes()
        for key, value in (
            ("level_duration", 9), ("level_duration", 601),
            ("level_duration", 30.5), ("level_difficulty", 0),
            ("level_difficulty", 6), ("super_blink_gain", 0.049),
            ("super_blink_gain", 1.01), ("super_blink_gain", True),
            ("super_decay_delay", 0.49), ("super_decay_delay", 15.01),
            ("super_decay_rate", -0.01), ("super_decay_rate", 0.51),
        ):
            with self.subTest(key=key, value=value):
                response = self.client.post("/config", json={key: value, "hand_gain": 2.5})
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
                self.assertEqual(self.cfg.hand_gain, settings.Config().hand_gain)
                self.assertEqual(self.config_file.read_bytes(), before)

    def test_json_mutation_routes_reject_non_object_payloads(self):
        for route in ("/config", "/score"):
            for body in ("[]", '"invalid"', "null", "{"):
                with self.subTest(route=route, body=body):
                    response = self.client.post(route, data=body, content_type="application/json")
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.get_json())
        self.assertFalse(self.config_file.exists())

    def test_score_persists_only_the_best_result(self):
        self.assertEqual(self.client.post("/score", json={"score": 42}).get_json(),
                         {"best_score": 42})
        before = self.config_file.read_bytes()
        for score in (9, 42, -100, "invalid", None):
            with self.subTest(score=score):
                response = self.client.post("/score", json={"score": score})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {"best_score": 42})
                self.assertEqual(self.config_file.read_bytes(), before)
        self.assertEqual(self.client.post("/score", json={"score": 80}).get_json(),
                         {"best_score": 80})
        restored = settings.Config()
        restored.load()
        self.assertEqual(restored.best_score, 80)

    def test_calibration_start_and_reset_routes(self):
        response = self.client.post("/calibrate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True, "frames": 25})
        self.assertEqual(self.tracker._calib_samples, [])
        self.cfg.gaze_neutral_x = 0.2
        self.cfg.gaze_neutral_y = -0.1
        response = self.client.post("/calibrate/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"ok": True})
        self.assertIsNone(self.tracker._calib_samples)
        self.assertFalse(self.tracker.get_state()["calibrating"])
        saved = json.loads(self.config_file.read_text(encoding="utf-8"))
        self.assertEqual((saved["gaze_neutral_x"], saved["gaze_neutral_y"]), (0, 0))

    def test_camera_start_reports_capture_failure(self):
        captures = [Mock(isOpened=Mock(return_value=False)) for _ in range(2)]
        with patch.object(game.cv2, "VideoCapture", side_effect=captures):
            response = self.client.post("/camera/start")
        self.assertEqual(response.status_code, 500)
        body = response.get_json()
        self.assertFalse(body["ok"])
        self.assertTrue(body["error"])
        self.assertFalse(self.tracker.running)
        self.assertFalse(self.client.get("/state").get_json()["camera_open"])
        for capture in captures:
            capture.release.assert_called_once()

    def test_camera_start_stop_succeeds_without_a_physical_device(self):
        capture = Mock(isOpened=Mock(return_value=True))
        worker = Mock(is_alive=Mock(return_value=False))
        with patch.object(game.cv2, "VideoCapture", return_value=capture) as open_capture, \
                patch.object(game.threading, "Thread", return_value=worker):
            response = self.client.post("/camera/start")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["ok"])
            self.assertTrue(self.tracker.running)
            self.assertTrue(self.client.get("/state").get_json()["camera_open"])
            worker.start.assert_called_once()
            worker.is_alive.return_value = True
            self.assertEqual(self.client.post("/camera/start").status_code, 200)
            worker.start.assert_called_once()
            open_capture.assert_called_once()
            worker.is_alive.return_value = False
            response = self.client.post("/camera/stop")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        self.assertFalse(self.tracker.running)
        self.assertFalse(self.tracker.get_state()["camera_open"])
        capture.release.assert_called_once()
        worker.join.assert_called_once()

    def test_video_feed_produces_a_complete_mjpeg_chunk(self):
        encoded = Mock(tobytes=Mock(return_value=b"\xff\xd8test-jpeg\xff\xd9"))
        frame = object()
        with patch.object(self.tracker, "get_frame", return_value=frame), \
                patch.object(game.cv2, "imencode", return_value=(True, encoded)) as encode:
            response = self.client.get("/video_feed", buffered=False)
            try:
                self.assertEqual(response.status_code, 200)
                self.assertIn("multipart/x-mixed-replace", response.content_type)
                self.assertIn("boundary=frame", response.content_type)
                chunk = next(iter(response.response))
                self.assertEqual(chunk, b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                 b"\xff\xd8test-jpeg\xff\xd9\r\n")
                encode.assert_called_once_with(".jpg", frame)
            finally:
                response.close()


class GestureTests(SmileGunTestCase):
    def test_actual_mediapipe_indices_pair_each_iris_with_its_own_eye(self):
        state = self.process(face=actual_mediapipe_face())
        self.assertAlmostEqual(state["iris_x"], 0.0)
        self.assertAlmostEqual(state["iris_y"], 0.0)
        self.assertEqual(state["super_charge"], 0.0)
        for _ in range(3):
            self.process(face=actual_mediapipe_face())
        self.assertEqual(self.tracker.get_state()["super_charge"], 0.0)
        state = self.process(face=actual_mediapipe_face(closed=True))
        self.assertEqual(state["super_charge"], 0.0)
        state = self.process(face=actual_mediapipe_face())
        self.assertEqual(state["super_charge"], 0.25)
        self.assertTrue(state["charging"])
        state = self.process(face=actual_mediapipe_face(), elapsed=0.25)
        self.assertEqual(state["super_charge"], 0.25)
        self.assertFalse(state["charging"])

    def test_fist_fires_once_until_the_palm_releases(self):
        state = self.process(hand=hand_landmarks(openness=1.5))
        self.assertEqual(state["shot_id"], 0)
        self.assertFalse(state["fist"])
        state = self.process(hand=hand_landmarks(openness=0.7))
        self.assertEqual(state["shot_id"], 1)
        self.assertTrue(state["fist"])
        for openness in (0.7, 1.1, 0.8, 1.2):
            state = self.process(hand=hand_landmarks(openness=openness))
            self.assertEqual(state["shot_id"], 1)
            self.assertTrue(state["fist"])
        self.process(hand=hand_landmarks(openness=1.5))
        state = self.process(hand=hand_landmarks(openness=0.7))
        self.assertEqual(state["shot_id"], 2)
        self.assertFalse(state["face"])

    def test_palm_aim_is_mirrored_and_clamped_to_field(self):
        self.cfg.hand_gain = 1.0
        state = self.process(hand=hand_landmarks(x=0.25, y=0.7))
        self.assertAlmostEqual(state["aim_x"], 0.75)
        self.assertAlmostEqual(state["aim_y"], 0.7)
        self.cfg.hand_gain = 2.0
        state = self.process(hand=hand_landmarks(x=0.1, y=0.9))
        self.assertEqual((state["aim_x"], state["aim_y"]), (1.0, 1.0))
        state = self.process(hand=hand_landmarks(x=0.9, y=0.1))
        self.assertEqual((state["aim_x"], state["aim_y"]), (0.0, 0.0))
        absent = self.process()
        self.assertFalse(absent["hand_present"])
        self.assertEqual((absent["aim_x"], absent["aim_y"]), (0.0, 0.0))

    def test_gaze_alone_and_held_closed_eyes_do_not_charge(self):
        for gaze in (-0.3, 0.3, -0.2, 0.2):
            state = self.process(face=face_landmarks(gaze_x=gaze), elapsed=1.0)
            self.assertEqual(state["super_charge"], 0.0)
            self.assertEqual(state["super_id"], 0)
            self.assertEqual(state["blink_id"], 0)
        for _ in range(8):
            state = self.process(face=face_landmarks(closed=True))
            self.assertEqual(state["super_charge"], 0.0)
            self.assertEqual(state["blink_id"], 0)
        state = self.process(face=face_landmarks())
        self.assertEqual(state["super_charge"], 0.25)
        self.assertEqual(state["super_id"], 0)
        self.assertEqual(state["blink_id"], 1)
        self.process(face=face_landmarks(), elapsed=0.5)
        self.assertEqual(self.tracker.get_state()["super_charge"], 0.25)

    def test_four_completed_blinks_fire_once_and_preserve_charge_over_pause(self):
        self.cfg.super_decay_rate = 0.0
        for number in (1, 2, 3):
            state = self.blink()
            self.assertEqual(state["super_charge"], number / 4)
            self.assertEqual(state["super_id"], 0)
        state = self.process(face=face_landmarks(), elapsed=20.0)
        self.assertEqual(state["super_charge"], 0.75)
        state = self.blink()
        self.assertEqual(state["super_id"], 1)
        self.assertEqual(state["super_charge"], 0.0)
        self.assertFalse(state["super_ready"])
        for _ in range(3):
            state = self.process(face=face_landmarks())
            self.assertEqual(state["super_id"], 1)
            self.assertEqual(state["super_charge"], 0.0)
        state = self.blink()
        self.assertEqual(state["super_charge"], 0.25)
        self.assertEqual(state["super_id"], 1)

    def test_face_loss_does_not_complete_an_interrupted_blink(self):
        self.blink()
        self.process(face=face_landmarks(closed=True))
        state = self.process(elapsed=2.0)
        self.assertFalse(state["face"])
        self.assertEqual(state["super_charge"], 0.25)
        state = self.process(face=face_landmarks())
        self.assertEqual(state["super_charge"], 0.25)

    def test_arbitrary_blink_gain_accumulates_until_threshold(self):
        self.client.post("/config", json={"super_blink_gain": 0.3})
        for number in range(1, 4):
            state = self.blink()
            self.assertAlmostEqual(state["super_charge"], number * 0.3)
            self.assertEqual(state["blink_id"], number)
            self.assertEqual(state["super_id"], 0)
        state = self.blink()
        self.assertEqual(state["super_id"], 1)
        self.assertEqual(state["super_charge"], 0)
        self.assertEqual(state["blink_id"], 4)

    def test_changing_blink_gain_preserves_existing_energy(self):
        self.client.post("/config", json={"super_blink_gain": 0.3})
        self.assertAlmostEqual(self.blink()["super_charge"], 0.3)
        self.client.post("/config", json={"super_blink_gain": 0.2})
        self.assertAlmostEqual(self.client.get("/state").get_json()["super_charge"], 0.3)
        self.assertAlmostEqual(self.blink()["super_charge"], 0.5)
        self.client.post("/config", json={"super_blink_gain": 0.5})
        state = self.blink()
        self.assertEqual(state["super_charge"], 0)
        self.assertEqual(state["super_id"], 1)
        self.assertEqual(state["blink_id"], 3)

    def test_smallest_gain_fires_on_twentieth_blink_despite_rounding(self):
        self.cfg.super_blink_gain = 0.05
        self.cfg.super_decay_rate = 0.0
        for number in range(1, 20):
            state = self.blink()
            self.assertAlmostEqual(state["super_charge"], number * 0.05)
            self.assertEqual(state["super_id"], 0)
        state = self.blink()
        self.assertEqual(state["super_id"], 1)
        self.assertEqual(state["super_charge"], 0)
        self.assertEqual(state["blink_id"], 20)

    def test_decay_grace_and_state_polling_work_without_a_face(self):
        self.blink()
        state = self.blink()
        self.assertAlmostEqual(state["super_charge"], 0.5)
        state = self.process(elapsed=2.999)
        self.assertFalse(state["face"])
        self.assertAlmostEqual(state["super_charge"], 0.5)
        self.clock += 0.001
        state = self.client.get("/state").get_json()
        self.assertAlmostEqual(state["super_charge"], 0.5)
        self.clock += 2.0
        state = self.client.get("/state").get_json()
        self.assertAlmostEqual(state["super_charge"], 0.34)
        self.assertFalse(state["face"])
        self.assertEqual(state["blink_id"], 2)
        self.assertAlmostEqual(self.client.get("/state").get_json()["super_charge"], 0.34)
        self.clock += 20.0
        self.assertEqual(self.client.get("/state").get_json()["super_charge"], 0.0)

    def test_decay_depends_on_elapsed_time_instead_of_poll_frequency(self):
        def charged_tracker():
            self.clock = 100.0
            tracker = game.CameraTracker()
            tracker._detect_hand = Mock(return_value=None)
            tracker._make_mesh = Mock()
            for closed in (True, False, True, False):
                self.clock += 0.05
                tracker._make_mesh.return_value = actual_mediapipe_face(closed)
                tracker._process_frame(object())
            tracker._make_mesh.return_value = None
            tracker._process_frame(object())
            self.assertAlmostEqual(tracker.get_state()["super_charge"], 0.5)
            return tracker, self.clock

        sparse, last_blink = charged_tracker()
        self.clock = last_blink + 5.0
        sparse_charge = sparse.get_state()["super_charge"]
        frequent, last_blink = charged_tracker()
        for step in range(1, 501):
            self.clock = last_blink + step * 0.01
            frequent_charge = frequent.get_state()["super_charge"]
        self.assertAlmostEqual(sparse_charge, 0.34)
        self.assertAlmostEqual(frequent_charge, sparse_charge)

    def test_blink_adds_gain_to_decayed_energy_and_restarts_grace_period(self):
        self.blink()
        self.blink()
        self.clock += 5.0
        self.assertAlmostEqual(self.client.get("/state").get_json()["super_charge"], 0.34)
        self.client.post("/config", json={"super_blink_gain": 0.3})
        state = self.blink()
        # Another 0.1 second passes during closing and reopening before the gain.
        self.assertAlmostEqual(state["super_charge"], 0.34 - 0.1 * 0.08 + 0.3)
        self.assertEqual(state["blink_id"], 3)
        self.clock += 2.9
        self.assertAlmostEqual(self.client.get("/state").get_json()["super_charge"], 0.632)

    def test_decay_can_require_an_extra_blink_before_super(self):
        self.cfg.super_blink_gain = 0.35
        self.blink()
        self.blink()
        self.clock += 5.0
        self.assertAlmostEqual(self.tracker.get_state()["super_charge"], 0.54)
        state = self.blink()
        self.assertAlmostEqual(state["super_charge"], 0.882)
        self.assertEqual(state["super_id"], 0)
        state = self.blink()
        self.assertEqual(state["super_id"], 1)
        self.assertEqual(state["blink_id"], 4)
        self.assertEqual(state["super_charge"], 0.0)

    def test_calibration_collects_25_open_eye_frames_and_rejects_outlier(self):
        self.client.post("/calibrate")
        for _ in range(10):
            self.process(face=face_landmarks(gaze_x=3, gaze_y=-2, closed=True))
        self.assertEqual(self.tracker._calib_samples, [])
        samples = [(0.12 + 0.004 * (i % 4 - 1.5),
                    -0.08 - 0.004 * (i % 4 - 1.5)) for i in range(24)]
        for gaze_x, gaze_y in samples[:-1] + [(4.0, -3.0)]:
            self.process(face=face_landmarks(gaze_x, gaze_y))
        self.assertEqual(len(self.tracker._calib_samples), 24)
        self.assertFalse(self.config_file.exists())
        self.assertTrue(self.tracker.get_state()["calibrating"])
        self.process(face=face_landmarks(*samples[-1]))
        self.assertIsNone(self.tracker._calib_samples)
        self.assertFalse(self.tracker.get_state()["calibrating"])
        self.assertAlmostEqual(self.cfg.gaze_neutral_x, 0.12)
        self.assertAlmostEqual(self.cfg.gaze_neutral_y, -0.08)
        restored = settings.Config()
        restored.load()
        self.assertAlmostEqual(restored.gaze_neutral_x, 0.12)
        self.assertAlmostEqual(restored.gaze_neutral_y, -0.08)


class CameraLifecycleTests(SmileGunTestCase):
    def interrupt_tracking(self, processing_error=False):
        """Inspect recovery before explicit shutdown intentionally resets it."""
        self.blink()
        self.client.post("/calibrate")
        self.process(face=face_landmarks(gaze_x=0.1, gaze_y=0.02))
        self.process(hand=hand_landmarks(openness=0.7),
                     face=face_landmarks(closed=True))
        samples = list(self.tracker._calib_samples)
        self.tracker.frame = object()
        self.tracker.running = True
        capture = Mock(isOpened=Mock(return_value=True))
        observed = {}
        reads = 0

        def read():
            nonlocal reads
            reads += 1
            if reads == 1:
                return (True, object()) if processing_error else (False, None)
            observed["state"] = self.tracker.get_state()
            observed["samples"] = self.tracker._calib_samples
            observed["frame"] = self.tracker.frame
            observed["previous_blink"] = self.tracker._prev_blink
            self.tracker._stop_event.set()
            return False, None

        capture.read.side_effect = read
        self.tracker.cap = capture
        with patch.object(self.tracker, "_process_frame",
                          side_effect=ValueError("bad landmarks")):
            self.tracker._loop()
        state = observed["state"]
        self.assertEqual(state["super_charge"], 0.25)
        self.assertEqual(state["blink_id"], 1)
        self.assertTrue(state["calibrating"])
        self.assertEqual(observed["samples"], samples)
        self.assertIsNone(observed["frame"])
        self.assertFalse(observed["previous_blink"])
        self.assertFalse(state["hand_present"])
        self.assertFalse(state["fist"])
        self.assertFalse(state["face"])
        capture.release.assert_called_once()
        return state

    def test_temporary_read_failure_keeps_power_and_calibration_progress(self):
        self.interrupt_tracking()

    def test_frame_processing_failure_keeps_power_and_calibration_progress(self):
        state = self.interrupt_tracking(processing_error=True)
        self.assertEqual(state["error"], "ValueError: bad landmarks")

    def test_stop_timeout_reports_error_and_blocks_second_worker(self):
        worker = Mock(is_alive=Mock(return_value=True))
        capture = Mock()
        self.tracker.thread = worker
        self.tracker.cap = capture
        self.tracker.running = True
        response = self.client.post("/camera/stop")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["ok"])
        self.assertTrue(response.get_json()["error"])
        worker.join.assert_called_once_with(timeout=self.tracker.STOP_TIMEOUT)
        capture.release.assert_not_called()
        self.assertFalse(self.tracker.running)
        with patch.object(game.cv2, "VideoCapture") as open_capture:
            response = self.client.post("/camera/start")
        self.assertEqual(response.status_code, 500)
        open_capture.assert_not_called()

    def test_camera_loop_releases_capture_after_stop_during_read(self):
        frame = object()
        capture = Mock(isOpened=Mock(return_value=True))
        reads = iter(((True, frame), (False, None)))

        def read():
            result = next(reads)
            if not result[0]:
                self.tracker._stop_event.set()
            return result

        capture.read.side_effect = read
        self.tracker.cap = capture
        self.tracker.running = True
        with patch.object(self.tracker, "_process_frame") as process_frame:
            self.tracker._loop()
        process_frame.assert_called_once_with(frame)
        self.assertEqual(self.tracker.get_state()["frames_read"], 1)
        capture.release.assert_called_once()
        self.assertIsNone(self.tracker.cap)
        self.assertFalse(self.tracker.running)

    def test_stop_allows_worker_to_finish_frame_under_state_lock(self):
        """Joining while holding tracker.lock deadlocks this actual worker."""
        started = threading.Event()
        finished = threading.Event()
        pause = threading.Event()
        self.tracker.running = True
        self.tracker.cap = Mock()

        def finish_in_flight_frame():
            started.set()
            while self.tracker.running:
                pause.wait(0.001)
            with self.tracker.lock:
                finished.set()

        worker = threading.Thread(target=finish_in_flight_frame, daemon=True)
        self.tracker.thread = worker
        worker.start()
        self.assertTrue(started.wait(1.0))
        failures = []

        def stop():
            try:
                self.tracker.stop()
            except Exception as error:
                failures.append(error)

        stopper = threading.Thread(target=stop, daemon=True)
        stopper.start()
        stopper.join(timeout=1.5)
        self.assertFalse(stopper.is_alive(), "stop waited for worker while retaining its lock")
        self.assertFalse(failures, repr(failures))
        self.assertTrue(finished.is_set())
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.tracker.get_state()["camera_open"])


if __name__ == "__main__":
    unittest.main()
