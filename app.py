import os
import threading
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, request, jsonify
from config import CFG
from diagnostics import suppress_noise

# Константы для landmarks
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_OUTER = 362
LEFT_EYE_INNER = 263
RIGHT_EYE_OUTER = 130
RIGHT_EYE_INNER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
NOSE_TIP = 4
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
BOTTOM_LIP = 14

MIN_EYE_WIDTH = 0.012

# Класс для экспоненциального скользящего среднего
class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# Функция для измерения координат
def measure(lm):
    return lm.x, lm.y

# Функция для расчета прицела
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

# Класс для отслеживания камеры
class CameraTracker:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.frame = None
        self.state = {
            "camera": False,
            "aim_x": 0.5,
            "aim_y": 0.5,
            "iris_x": 0.5,
            "iris_y": 0.5,
            "smile_score": 0.0,
            "mouth_open_score": 0.0,
            "smile": False,
            "shot_id": 0,
        }
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.iris_ema_x = EMA(CFG.smoothing_alpha)
        self.iris_ema_y = EMA(CFG.smoothing_alpha)
        self.head_ema_x = EMA(CFG.smoothing_alpha)
        self.head_ema_y = EMA(CFG.smoothing_alpha)
        self.smile_ema = EMA(CFG.smoothing_alpha)
        self.mouth_ema = EMA(CFG.smoothing_alpha)

    def _make_mesh(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def start(self):
        with self.lock:
            if self.running:
                return
            self.cap = cv2.VideoCapture(CFG.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(CFG.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
            self.running = True
            self.thread = threading.Thread(target=self._loop)
            self.thread.start()

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.cap.release()
            self.thread.join()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
                self._process_frame(frame)

    def _process_frame(self, frame):
        mesh = self._make_mesh(frame)
        if mesh:
            # Координаты зрачков
            iris_lx, iris_ly = measure(mesh.landmark[LEFT_IRIS_CENTER])
            iris_rx, iris_ry = measure(mesh.landmark[RIGHT_IRIS_CENTER])
            iris_x = (iris_lx + iris_rx) / 2
            iris_y = (iris_ly + iris_ry) / 2

            # Координаты глаз
            eye_lx1, eye_ly1 = measure(mesh.landmark[LEFT_EYE_OUTER])
            eye_lx2, eye_ly2 = measure(mesh.landmark[LEFT_EYE_INNER])
            eye_rx1, eye_ry1 = measure(mesh.landmark[RIGHT_EYE_OUTER])
            eye_rx2, eye_ry2 = measure(mesh.landmark[RIGHT_EYE_INNER])

            # Ширина глаз
            eye_lw = abs(eye_lx1 - eye_lx2)
            eye_rw = abs(eye_rx1 - eye_rx2)

            # Нормализация координат зрачков
            iris_x = (iris_x - 0.5) / eye_lw
            iris_y = (iris_y - 0.5) / eye_lw

            # Нормализация координат головы
            nose_x, nose_y = measure(mesh.landmark[NOSE_TIP])
            head_x = nose_x - 0.5
            head_y = nose_y - 0.5

            # Нормализация координат улыбки
            mouth_lx, mouth_ly = measure(mesh.landmark[MOUTH_LEFT])
            mouth_rx, mouth_ry = measure(mesh.landmark[MOUTH_RIGHT])
            mouth_x = (mouth_lx + mouth_rx) / 2
            mouth_y = (mouth_ly + mouth_ry) / 2
            smile_score = mouth_y - nose_y
            mouth_open_score = abs(mouth_lx - mouth_rx)

            # Сглаживание
            iris_x = self.iris_ema_x.update(iris_x)
            iris_y = self.iris_ema_y.update(iris_y)
            head_x = self.head_ema_x.update(head_x)
            head_y = self.head_ema_y.update(head_y)
            smile_score = self.smile_ema.update(smile_score)
            mouth_open_score = self.mouth_ema.update(mouth_open_score)

            # Проверка улыбки
            smile = smile_score > CFG.smile_threshold and mouth_open_score < CFG.smile_hysteresis * CFG.smile_threshold

            # Расчет прицела
            aim_x, aim_y = hybrid_aim(head_x, head_y, iris_x, iris_y, 0.5, 0.5)

            # Обновление состояния
            self.state.update({
                "camera": True,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "iris_x": iris_x,
                "iris_y": iris_y,
                "smile_score": smile_score,
                "mouth_open_score": mouth_open_score,
                "smile": smile,
                "shot_id": self.state["shot_id"] + (1 if smile else 0),
            })

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_state(self):
        with self.lock:
            return self.state

# Инициализация трекера
tracker = CameraTracker()

# Создание Flask приложения
app = Flask(__name__)

# MJPEG поток
def mjpeg_stream():
    while True:
        frame = tracker.get_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Главная страница
@app.route('/')
def index():
    return app.send_static_file('index.html')

# MJPEG поток для превью камеры
@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Получение состояния
@app.route('/state')
def state():
    return jsonify(tracker.get_state())

# Получение/установка конфигурации
@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        data = request.json
        CFG.update(**data)
        return jsonify(CFG.as_dict())
    return jsonify(CFG.as_dict())

# Запуск камеры
@app.route('/camera/start', methods=['POST'])
def camera_start():
    try:
        tracker.start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Остановка камеры
@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    tracker.stop()
    return jsonify({"ok": True})

# Запуск приложения
if __name__ == "__main__":
    suppress_noise()
    tracker.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
