import threading
import time
import cv2
import mediapipe as mp
from flask import Flask, Response, request, jsonify, render_template
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
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LIP_TOP_INNER = 13
LIP_BOTTOM_INNER = 14

# Глаз уже этой ширины (в долях кадра) считаем невалидным: на него нельзя делить.
MIN_EYE_WIDTH = 0.012

# Центр игрового поля: прицел уходит на клиент в долях поля (0..1).
FIELD_CENTER = 0.5

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

def clamp01(value: float) -> float:
    """Прицел не должен выходить за поле."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)

def eye_gaze(iris_x: float, iris_y: float, outer_x: float, inner_x: float,
             top_y: float, bottom_y: float) -> tuple[float, float, float]:
    """Взгляд одного глаза в единицах размера ЭТОГО глаза.

    Зрачок нормируется на центр своего глаза, а не на центр кадра. Когда
    человек двигает головой, глаз и зрачок едут по кадру вместе, и отношение
    (зрачок - центр глаза) / ширина глаза не меняется — остаётся только
    поворот зрачка. Именно поэтому кубик слушается только глаз.

    Третий элемент результата — ширина глаза: по ней нормируются метрики рта,
    чтобы улыбка не зависела от того, близко или далеко лицо от камеры.
    """
    eye_w = max(abs(outer_x - inner_x), MIN_EYE_WIDTH)
    eye_h = max(abs(top_y - bottom_y), MIN_EYE_WIDTH)
    center_x = (outer_x + inner_x) / 2.0
    center_y = (top_y + bottom_y) / 2.0
    return (iris_x - center_x) / eye_w, (iris_y - center_y) / eye_h, eye_w

def gaze_to_aim(gaze_x: float, gaze_y: float) -> tuple[float, float]:
    """Прицел только по зрачкам: смещение взгляда от откалиброванной нейтрали.

    CFG.gaze_neutral_x/y запоминает кнопка «Калибровка», когда человек смотрит
    в центр поля. Поэтому прицел не зависит ни от посадки, ни от размера лица,
    ни от положения головы — только от движения зрачков.
    """
    offset_x = (gaze_x - CFG.gaze_neutral_x) * CFG.iris_gain
    offset_y = (gaze_y - CFG.gaze_neutral_y) * CFG.iris_gain * CFG.iris_gain_y
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
            "iris_x": 0.0,
            "iris_y": 0.0,
            "gaze_dx": 0.0,
            "gaze_dy": 0.0,
            "neutral_x": CFG.gaze_neutral_x,
            "neutral_y": CFG.gaze_neutral_y,
            "smile_score": 0.0,
            "mouth_open_score": 0.0,
            "smile": False,
            "mouth_open": False,
            "calibrating": False,
            "shot_id": 0,
            # Диагностика камеры: без неё «НЕТ КАМЕРЫ» не отличить от
            # «кадры идут, но лица не видно» и от «устройство занято».
            "frames_read": 0,
            "face": False,
            "fps": 0.0,
            "camera_open": False,
            "error": None,
        }
        self._calib_samples = None
        self._prev_smile = False
        self._prev_mouth_open = False
        self._held = {"smile": False, "mouth_open": False}
        self._fps_window_start = time.time()
        self._fps_frames = 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.iris_ema_x = EMA(CFG.smoothing_alpha)
        self.iris_ema_y = EMA(CFG.smoothing_alpha)
        self.smile_ema = EMA(CFG.smoothing_alpha)
        self.mouth_ema = EMA(CFG.smoothing_alpha)

    CALIB_FRAMES = 15
    MAX_READ_FAILURES = 60      # ~2 секунды без кадров
    REOPEN_COOLDOWN = 5.0       # сек между попытками переоткрыть камеру

    def start_calibration(self) -> None:
        """Запомнить нейтраль взгляда: усредняем следующие CALIB_FRAMES кадров."""
        with self.lock:
            self._calib_samples = []

    def _collect_calibration(self, gaze_x: float, gaze_y: float) -> None:
        if self._calib_samples is None:
            return
        self._calib_samples.append((gaze_x, gaze_y))
        if len(self._calib_samples) < self.CALIB_FRAMES:
            return
        xs = [sample[0] for sample in self._calib_samples]
        ys = [sample[1] for sample in self._calib_samples]
        CFG.gaze_neutral_x = sum(xs) / len(xs)
        CFG.gaze_neutral_y = sum(ys) / len(ys)
        self._calib_samples = None

    def _threshold(self, key: str, value: float, threshold: float) -> bool:
        """Порог с гистерезисом: включается выше threshold, гаснет ниже
        threshold * smile_hysteresis.

        Без гистерезиса значение, дрожащее у порога, выдаёт очередь выстрелов.
        """
        release = threshold * CFG.smile_hysteresis
        if self._held[key]:
            if value < release:
                self._held[key] = False
        elif value > threshold:
            self._held[key] = True
        return self._held[key]

    def _make_mesh(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def _open_capture(self) -> None:
        """Открыть камеру: сначала DSHOW, потом бэкенд по умолчанию."""
        self.cap = cv2.VideoCapture(CFG.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(CFG.camera_index)

    def start(self):
        with self.lock:
            if self.running:
                return
            # После перезапуска камеры стрельба начинается с чистого состояния.
            self._prev_smile = False
            self._prev_mouth_open = False
            self._held = {"smile": False, "mouth_open": False}
            self._open_capture()
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
            self.state["camera_open"] = True
            self.state["error"] = None
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
            self.frame = None
            self.state["camera"] = False
            self.state["face"] = False
            self.state["camera_open"] = False

    def _loop(self):
        read_failures = 0
        last_reopen = 0.0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Камера открыта, но кадров нет: так выглядит занятое устройство
                # (или кривой бэкенд). Не крутимся вхолостую и говорим об этом.
                read_failures += 1
                if read_failures >= self.MAX_READ_FAILURES:
                    if time.time() - last_reopen > self.REOPEN_COOLDOWN:
                        last_reopen = time.time()
                        read_failures = 0
                        self._open_capture()
                        with self.lock:
                            self.state["camera_open"] = self.cap.isOpened()
                    with self.lock:
                        # Кадров нет — значит и про лицо мы ничего не знаем:
                        # не оставляем устаревшее «лицо есть».
                        self.state["face"] = False
                        self.state["camera"] = False
                        self.state["error"] = (
                            "Камера открыта, но не отдаёт кадры. Устройство занято "
                            "другим процессом (или камера выключена)."
                        )
                time.sleep(0.02)
                continue
            read_failures = 0
            with self.lock:
                self.frame = frame
                self.state["error"] = None
                self.state["frames_read"] += 1
                self._fps_frames += 1
                now = time.time()
                if now - self._fps_window_start >= 1.0:
                    self.state["fps"] = self._fps_frames / (now - self._fps_window_start)
                    self._fps_window_start = now
                    self._fps_frames = 0
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    # Один плохой кадр не должен убивать поток трекинга.
                    self.state["error"] = f"{type(exc).__name__}: {exc}"

    def _process_frame(self, frame):
        mesh = self._make_mesh(frame)
        self.state["face"] = bool(mesh)
        if not mesh:
            return

        lm = mesh.landmark

        # Взгляд: зрачок относительно центра своего глаза, в размерах глаза
        gaze_lx, gaze_ly, eye_lw = eye_gaze(lm[LEFT_IRIS_CENTER].x, lm[LEFT_IRIS_CENTER].y,
                                            lm[LEFT_EYE_OUTER].x, lm[LEFT_EYE_INNER].x,
                                            lm[LEFT_EYE_TOP].y, lm[LEFT_EYE_BOTTOM].y)
        gaze_rx, gaze_ry, eye_rw = eye_gaze(lm[RIGHT_IRIS_CENTER].x, lm[RIGHT_IRIS_CENTER].y,
                                            lm[RIGHT_EYE_OUTER].x, lm[RIGHT_EYE_INNER].x,
                                            lm[RIGHT_EYE_TOP].y, lm[RIGHT_EYE_BOTTOM].y)
        gaze_x = self.iris_ema_x.update((gaze_lx + gaze_rx) / 2.0)
        gaze_y = self.iris_ema_y.update((gaze_ly + gaze_ry) / 2.0)
        eye_w = (eye_lw + eye_rw) / 2.0     # масштаб лица для метрик рта

        # Улыбка — это рот, растянутый в ширину (губы можно сжать, рот
        # открывать не надо), поэтому меряем ширину рта, а не подъём углов.
        mouth_w = abs(lm[MOUTH_LEFT].x - lm[MOUTH_RIGHT].x)
        smile_score = self.smile_ema.update(mouth_w / eye_w)

        # Широко открытый рот — просвет между внутренними губами.
        mouth_h = abs(lm[LIP_TOP_INNER].y - lm[LIP_BOTTOM_INNER].y)
        mouth_open_score = self.mouth_ema.update(mouth_h / eye_w)

        smile = self._threshold("smile", smile_score, CFG.smile_threshold)
        mouth_open = self._threshold("mouth_open", mouth_open_score,
                                     CFG.mouth_open_threshold)

        # Выстрел по началу события: улыбка — один, открытый рот — двойной.
        shots = 0
        if smile and not self._prev_smile:
            shots = 1
        if mouth_open and not self._prev_mouth_open:
            shots = 2
        self._prev_smile = smile
        self._prev_mouth_open = mouth_open

        self._collect_calibration(gaze_x, gaze_y)
        aim_x, aim_y = gaze_to_aim(gaze_x, gaze_y)

        # Обновление состояния
        self.state.update({
            "camera": True,
            "aim_x": aim_x,
            "aim_y": aim_y,
            "iris_x": gaze_x,
            "iris_y": gaze_y,
            "gaze_dx": gaze_x - CFG.gaze_neutral_x,
            "gaze_dy": gaze_y - CFG.gaze_neutral_y,
            "neutral_x": CFG.gaze_neutral_x,
            "neutral_y": CFG.gaze_neutral_y,
            "smile_score": smile_score,
            "mouth_open_score": mouth_open_score,
            "smile": smile,
            "mouth_open": mouth_open,
            "calibrating": self._calib_samples is not None,
            "shot_id": self.state["shot_id"] + shots,
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
            # Без паузы этот цикл выедает GIL: /state начинает отвечать по
            # 100 мс, и FPS в интерфейсе падает до 10.
            time.sleep(0.02)
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
    # Шаблон лежит в templates/index.html: статикой он не отдаётся.
    return render_template('index.html')

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

# Калибровка взгляда: запомнить текущее положение зрачков как нейтраль
@app.route('/calibrate', methods=['POST'])
def calibrate():
    tracker.start_calibration()
    return jsonify({"ok": True, "frames": tracker.CALIB_FRAMES})

# Сброс калибровки к нулевой нейтрали
@app.route('/calibrate/reset', methods=['POST'])
def calibrate_reset():
    CFG.gaze_neutral_x = 0.0
    CFG.gaze_neutral_y = 0.0
    return jsonify({"ok": True})

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
    try:
        tracker.start()
    except RuntimeError as e:
        # Сервер всё равно поднимаем: на странице есть кнопка запуска камеры.
        print(f"[!] {e}. Закрой приложение, которое держит камеру, и нажми "
              f"«ЗАПУСТИТЬ КАМЕРУ» на странице.")
    # use_reloader=False обязательно: с релоадером Flask поднимает ВТОРОЙ
    # процесс, tracker.start() выполняется в обоих, камеру открывают два
    # процесса — и тот, что получил устройство, не обслуживает браузер.
    # Снаружи это выглядит как чёрное превью и «НЕТ КАМЕРЫ».
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
