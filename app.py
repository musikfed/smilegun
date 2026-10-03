# smilegun/app.py
from __future__ import annotations

import os

# --- Тишина в консоли ---------------------------------------------------
# Переменные выставлены ДО импорта mediapipe: их читают TensorFlow Lite и
# связанные с ним библиотеки только в момент инициализации. Основную работу
# делает suppress_noise() ниже (он гасит строку на каждый HTTP-запрос).
#
# Проверено на mediapipe 0.10.21: эти переменные НЕ убирают строки
# "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." и "W0000 ...
# inference_feedback_manager.cc" — их пишет C++ напрямую в stderr, минуя
# логирование Python. Оставляем как безвредную страховку для других сборок:
# на некоторых версиях TensorFlow они действительно глушат INFO/WARNING.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # только фатальные
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, Response, request

from config import CFG
from diagnostics import print_report, suppress_noise

# --- Диагностика ДО всех тяжёлых операций -------------------------------
suppress_noise()   # первым делом: гасим служебный шум Flask/werkzeug
print_report()

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# --- Landmark IDs (MediaPipe FaceMesh, refine_landmarks=True) ----------
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
# Уголки глаз нужны, чтобы нормировать ход зрачка на ширину глаза:
# без этого смещение зрачка зависит от дистанции до камеры.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_LIP = 13
BOTTOM_LIP = 14

# Ниже этого не опускаем нормировку: при моргании/полуприкрытых глазах
# ширина глаза в кадре становится крошечной и усиление улетает в шум.
MIN_EYE_WIDTH = 0.012

# Прицел живёт в нормализованных координатах поля 0..1.
FIELD_CENTER = 0.5


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def deadzone(v: float, dz: float) -> float:
    """Мёртвая зона без скачка на границе: за зоной ход остаётся линейным."""
    if dz <= 0.0 or abs(v) <= dz:
        return 0.0
    return v - dz if v > 0.0 else v + dz


class EMA:
    """Экспоненциальное сглаживание — стабильнее и быстрее, чем deque-average."""
    __slots__ = ("alpha", "value")

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


def measure(lm) -> dict:
    """Сырые измерения кадра: поза головы и взгляд (без сглаживания).

    Возвращает нормализованные координаты кадра (голова, зрачки) и взгляд
    в единицах ширины глаза. Вынесено отдельно, чтобы формулу можно было
    проверять без камеры и MediaPipe.
    """
    # Голова: середина между щеками; по вертикали чуть подмешиваем нос —
    # он не так сильно реагирует на открывание рта, как линия щёк.
    head_x = (lm[LEFT_CHEEK].x + lm[RIGHT_CHEEK].x) * 0.5
    head_y = (0.8 * (lm[LEFT_CHEEK].y + lm[RIGHT_CHEEK].y) * 0.5
              + 0.2 * lm[NOSE_TIP].y)

    iris_x = (lm[LEFT_IRIS_CENTER].x + lm[RIGHT_IRIS_CENTER].x) * 0.5
    iris_y = (lm[LEFT_IRIS_CENTER].y + lm[RIGHT_IRIS_CENTER].y) * 0.5

    # Взгляд считается от центра глаза, а не от центра лица: расстояние
    # "зрачок минус середина щёк" почти не меняется при движении взгляда,
    # поэтому раньше зрачки не управляли прицелом. Нормируем на ширину
    # глаза, чтобы амплитуда не зависела от дистанции до камеры.
    eye_cx = (lm[LEFT_EYE_OUTER].x + lm[LEFT_EYE_INNER].x
              + lm[RIGHT_EYE_INNER].x + lm[RIGHT_EYE_OUTER].x) * 0.25
    eye_cy = (lm[LEFT_EYE_OUTER].y + lm[LEFT_EYE_INNER].y
              + lm[RIGHT_EYE_INNER].y + lm[RIGHT_EYE_OUTER].y) * 0.25
    eye_w = max(abs(lm[LEFT_EYE_INNER].x - lm[LEFT_EYE_OUTER].x),
                abs(lm[RIGHT_EYE_OUTER].x - lm[RIGHT_EYE_INNER].x),
                MIN_EYE_WIDTH)

    gaze_x = deadzone((iris_x - eye_cx) / eye_w, CFG.eye_deadzone)
    gaze_y = deadzone((iris_y - eye_cy) / eye_w, CFG.eye_deadzone)

    return {
        "head_x": head_x, "head_y": head_y,
        "iris_x": iris_x, "iris_y": iris_y,
        "gaze_x": gaze_x, "gaze_y": gaze_y,
    }


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
    offset_x = ((head_x - neutral_x) * CFG.head_gain
                + gaze_x * CFG.iris_gain)
    offset_y = ((head_y - neutral_y) * CFG.head_gain * CFG.head_gain_y
                + gaze_y * CFG.iris_gain * CFG.iris_gain_y)
    return clamp01(FIELD_CENTER + offset_x), clamp01(FIELD_CENTER + offset_y)


class CameraTracker:
    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.frame: bytes | None = None

        # Aim (нормализованные координаты поля 0..1)
        self.aim_x = 0.5
        self.aim_y = 0.5

        # Зрачки (нормализованные координаты кадра 0..1, для отладки)
        self.iris_x = 0.5
        self.iris_y = 0.5
        # Голова
        self.head_x = 0.5
        self.head_y = 0.5

        # Нейтраль головы: позиция, в которой прицел стоит в центре поля.
        # Медленно подстраивается под сидящего перед камерой человека.
        self.neutral_x = 0.5
        self.neutral_y = 0.5
        # Нормированный взгляд (в единицах ширины глаза) и его сглаженная версия
        self.gaze_x = 0.0
        self.gaze_y = 0.0

        # Улыбка
        self.smile_score = 0.0
        self.mouth_open_score = 0.0
        self.smile = False

        # Счётчик выстрелов (edge-триггер)
        self.shot_id = 0
        self.last_shot = 0.0
        self._smile_state = False

        # Сглаживание
        self.ema_ax = EMA(CFG.smoothing_alpha)
        self.ema_ay = EMA(CFG.smoothing_alpha)
        # Взгляд сглаживаем мягче: после нормировки на ширину глаза его
        # собственный шум MediaPipe усиливается сильнее, чем шум головы.
        self.ema_gx = EMA(CFG.smoothing_alpha * 0.6)
        self.ema_gy = EMA(CFG.smoothing_alpha * 0.6)

        self.face_mesh = self._make_mesh()

    @staticmethod
    def _make_mesh():
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ---------------- lifecycle ----------------
    def start(self):
        with self.lock:
            if self.running:
                return
            cap = cv2.VideoCapture(CFG.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(CFG.camera_index)
            if not cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.camera_height)
            cap.set(cv2.CAP_PROP_FPS, CFG.camera_fps)

            # Warm-up: DSHOW часто отдаёт чёрные первые кадры.
            for _ in range(5):
                cap.read()
                time.sleep(0.03)

            self.cap = cap
            self.face_mesh = self._make_mesh()
            self.ema_ax.reset()
            self.ema_ay.reset()
            self.ema_gx.reset()
            self.ema_gy.reset()
            self.neutral_x = 0.5
            self.neutral_y = 0.5
            self.gaze_x = 0.0
            self.gaze_y = 0.0
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            cap = self.cap
            self.cap = None
        if cap is not None:
            cap.release()

    # ---------------- main loop ----------------
    def _loop(self):
        while self.running:
            cap = self.cap
            if cap is None:
                time.sleep(0.02)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Зеркалим — глаз естественнее управляет "как в зеркале".
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                result = self.face_mesh.process(rgb)
            except Exception as e:
                # Не роняем поток из-за MediaPipe
                print(f"[tracker] face_mesh error: {e}")
                result = None

            now = time.monotonic()
            face_found = bool(result and result.multi_face_landmarks)
            smile_now = False

            if face_found:
                lm = result.multi_face_landmarks[0].landmark

                m = measure(lm)
                head_x, head_y = m["head_x"], m["head_y"]
                iris_x, iris_y = m["iris_x"], m["iris_y"]

                gx = self.ema_gx.update(m["gaze_x"])
                gy = self.ema_gy.update(m["gaze_y"])

                # ---- Автоподстройка нейтрали головы ----
                # Центр поля должен соответствовать естественной позе человека,
                # а не геометрическому центру кадра (0.5, 0.5): иначе остаётся
                # постоянный сдвиг. Двигаем нейтраль, только когда взгляд
                # спокоен и голова стоит близко к текущей нейтрали.
                rate = CFG.head_recenter_rate
                if rate > 0.0:
                    tol = CFG.head_recenter_tol
                    if abs(gx) < tol and abs(gy) < tol:
                        if abs(head_x - self.neutral_x) < 0.15:
                            self.neutral_x += rate * (head_x - self.neutral_x)
                        if abs(head_y - self.neutral_y) < 0.15:
                            self.neutral_y += rate * (head_y - self.neutral_y)
                self.neutral_x = min(0.75, max(0.25, self.neutral_x))
                self.neutral_y = min(0.75, max(0.25, self.neutral_y))

                raw_ax, raw_ay = hybrid_aim(head_x, head_y, gx, gy,
                                            self.neutral_x, self.neutral_y)

                with self.lock:
                    self.aim_x = self.ema_ax.update(raw_ax)
                    self.aim_y = self.ema_ay.update(raw_ay)
                    self.iris_x, self.iris_y = iris_x, iris_y
                    self.head_x, self.head_y = head_x, head_y
                    self.gaze_x, self.gaze_y = gx, gy

                # ---- Улыбка ----
                # width растёт при улыбке, height растёт при открытии рта.
                # Нормализуем на ширину лица, чтобы не зависеть от дистанции.
                face_width = abs(lm[RIGHT_CHEEK].x - lm[LEFT_CHEEK].x) + 1e-6
                mouth_w = abs(lm[RIGHT_MOUTH].x - lm[LEFT_MOUTH].x) / face_width
                mouth_h = abs(lm[BOTTOM_LIP].y - lm[TOP_LIP].y) / face_width

                # smile = растяжение минус штраф за открытие
                smile_score = mouth_w - 1.3 * mouth_h

                with self.lock:
                    self.smile_score = smile_score
                    self.mouth_open_score = mouth_h

                # Порог + гистерезис
                th_hi = CFG.smile_threshold
                th_lo = th_hi * CFG.smile_hysteresis
                with self.lock:
                    if not self._smile_state and smile_score > th_hi:
                        self._smile_state = True
                    elif self._smile_state and smile_score < th_lo:
                        self._smile_state = False
                    smile_now = self._smile_state

                # ---- Отрисовка ----
                for idx, color in (
                    (LEFT_IRIS_CENTER, (0, 255, 255)),
                    (RIGHT_IRIS_CENTER, (0, 255, 255)),
                    (LEFT_MOUTH, (255, 180, 0)),
                    (RIGHT_MOUTH, (255, 180, 0)),
                ):
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 4, color, -1)

                # Крестик прицела на превью
                ax = int(self.aim_x * w)
                ay = int(self.aim_y * h)
                cv2.drawMarker(frame, (ax, ay), (0, 0, 255),
                               cv2.MARKER_CROSS, 20, 2)

            # ---- Выстрел: edge + cooldown ----
            if smile_now and not self.smile and (now - self.last_shot) > CFG.fire_cooldown:
                self.shot_id += 1
                self.last_shot = now

            self.smile = smile_now

            # HUD
            cv2.putText(frame,
                        f"aim {self.aim_x:.2f},{self.aim_y:.2f}  "
                        f"head {self.head_x:.2f},{self.head_y:.2f}  "
                        f"gaze {self.gaze_x:+.2f},{self.gaze_y:+.2f}  "
                        f"smile {self.smile_score:+.2f}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "FIRE!" if self.smile else "READY",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 80, 255) if self.smile else (0, 220, 80),
                        2, cv2.LINE_AA)

            ok, encoded = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with self.lock:
                    self.frame = encoded.tobytes()

    # ---------------- accessors ----------------
    def get_frame(self):
        with self.lock:
            return self.frame

    def get_state(self):
        with self.lock:
            return {
                "aim_x": self.aim_x,
                "aim_y": self.aim_y,
                "iris_x": self.iris_x,
                "iris_y": self.iris_y,
                "head_x": self.head_x,
                "head_y": self.head_y,
                "gaze_x": self.gaze_x,
                "gaze_y": self.gaze_y,
                "neutral_x": self.neutral_x,
                "neutral_y": self.neutral_y,
                "smile": self.smile,
                "smile_score": self.smile_score,
                "mouth_open_score": self.mouth_open_score,
                "shot_id": self.shot_id,
                "camera": self.running,
            }


tracker = CameraTracker()


def mjpeg_stream():
    boundary = b"--frame\r\n"
    while True:
        frame = tracker.get_frame()
        if frame:
            yield (boundary
                   + b"Content-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        else:
            time.sleep(0.03)


# ---------------- routes ----------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def state():
    return jsonify(tracker.get_state())


@app.get("/config")
def get_config():
    return jsonify(CFG.as_dict())


@app.post("/config")
def set_config():
    data = request.get_json(force=True, silent=True) or {}
    try:
        CFG.update(**data)
        # применяем alpha к существующим EMA
        tracker.ema_ax.alpha = CFG.smoothing_alpha
        tracker.ema_ay.alpha = CFG.smoothing_alpha
        tracker.ema_gx.alpha = CFG.smoothing_alpha * 0.6
        tracker.ema_gy.alpha = CFG.smoothing_alpha * 0.6
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "config": CFG.as_dict()})


@app.post("/camera/start")
def camera_start():
    try:
        tracker.start()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


@app.post("/camera/stop")
def camera_stop():
    tracker.stop()
    return jsonify({"ok": True})


if __name__ == "__main__":
    try:
        tracker.start()
    except Exception as e:
        print(f"[!] Камера не запущена автоматически: {e}")
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        tracker.stop()