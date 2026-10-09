import atexit
import math
import os
import statistics
import threading
import time
import cv2
import mediapipe as mp
from flask import Flask, Response, request, jsonify, render_template
from config import CFG
from diagnostics import suppress_noise

# Константы для landmarks
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
LEFT_EYE_OUTER = 362
LEFT_EYE_INNER = 263
RIGHT_EYE_OUTER = 133
RIGHT_EYE_INNER = 33
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
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

# Адаптивный фильтр «One Euro»: плавный в покое (гасит дрожание) и отзывчивый
# при быстром движении. Стандарт для трекинга взгляда и руки.
class OneEuro:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.3,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, t: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        dx = self._alpha(self.d_cutoff, dt) * dx + \
            (1.0 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        self.dx_prev = dx
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self._alpha(cutoff, dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev

def clamp01(value: float) -> float:
    """Прицел не должен выходить за поле."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)

def eye_gaze(iris_x: float, iris_y: float, outer_x: float, inner_x: float,
             top_y: float, bottom_y: float) -> tuple[float, float, float]:
    """Взгляд одного глаза в единицах размера ЭТОГО глаза.

    Зрачок нормируется на центр своего глаза, а не на центр кадра. Когда
    человек двигает головой, глаз и зрачок едут по кадру вместе, и отношение
    (зрачок - центр глаза) / ширина глаза не меняется — остаётся только
    поворот зрачка. Это позволяет показывать взгляд в диагностике.

    Третий элемент результата — ширина глаза: по ней нормируются метрики рта,
    чтобы улыбка не зависела от того, близко или далеко лицо от камеры.
    """
    eye_w = max(abs(outer_x - inner_x), MIN_EYE_WIDTH)
    eye_h = max(abs(top_y - bottom_y), MIN_EYE_WIDTH)
    center_x = (outer_x + inner_x) / 2.0
    center_y = (top_y + bottom_y) / 2.0
    return (iris_x - center_x) / eye_w, (iris_y - center_y) / eye_h, eye_w

def apply_deadzone(value: float, deadzone: float) -> float:
    """Непрерывная мёртвая зона: 0 внутри, плавно наружу, без скачка на краю."""
    if deadzone <= 0:
        return value
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)

def gaze_to_aim(gaze_x: float, gaze_y: float) -> tuple[float, float]:
    """Старое преобразование взгляда; в текущей игре прицел управляется рукой."""
    dx = apply_deadzone(gaze_x - CFG.gaze_neutral_x, CFG.gaze_deadzone)
    dy = apply_deadzone(gaze_y - CFG.gaze_neutral_y, CFG.gaze_deadzone)
    offset_x = dx * CFG.iris_gain
    offset_y = dy * CFG.iris_gain * CFG.iris_gain_y
    return clamp01(FIELD_CENTER + offset_x), clamp01(FIELD_CENTER + offset_y)

# Класс для отслеживания камеры
class CameraTracker:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.thread = None
        self._closed = False
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
            "hand_present": False,
            "hand_x": 0.5,
            "hand_y": 0.5,
            "fist": False,
            "super_charge": 0.0,
            "super_ready": False,
            "super_id": 0,
            "blink_id": 0,
            "blink_count": 0,
            "charging": False,
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
        self._prev_fist = False
        self._fist_held = False
        self._prev_blink = False
        self._charge = 0.0
        self._charge_updated_at = time.monotonic()
        self._last_blink_time = None
        self._charging_until = 0.0
        self._held = {"smile": False, "mouth_open": False}
        self._fps_window_start = time.monotonic()
        self._fps_frames = 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
            max_num_faces=1
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        # One Euro вместо голого EMA: убирает дрожание взгляда и руки.
        self.gaze_flt_x = OneEuro(min_cutoff=1.2, beta=0.3)
        self.gaze_flt_y = OneEuro(min_cutoff=1.2, beta=0.3)
        self.hand_flt_x = OneEuro(min_cutoff=1.0, beta=0.2)
        self.hand_flt_y = OneEuro(min_cutoff=1.0, beta=0.2)
        self.smile_ema = EMA(CFG.smoothing_alpha)
        self.mouth_ema = EMA(CFG.smoothing_alpha)

    CALIB_FRAMES = 25
    BLINK_RATIO = 0.15          # раскрытие глаза ниже этого = моргание
    MAX_READ_FAILURES = 60      # ~2 секунды без кадров
    REOPEN_COOLDOWN = 5.0       # сек между попытками переоткрыть камеру
    STOP_TIMEOUT = 3.0         # не блокировать HTTP, если драйвер завис в read()
    CHARGE_PULSE_SECONDS = 0.2

    def _update_charge(self, now: float) -> None:
        """Интегрировать затухание один раз, независимо от кадров и опроса."""
        if self._last_blink_time is not None:
            decay_from = max(self._charge_updated_at,
                             self._last_blink_time + CFG.super_decay_delay)
            elapsed = max(0.0, now - decay_from)
            self._charge = max(0.0, self._charge - elapsed * CFG.super_decay_rate)
        self._charge_updated_at = now
        self.state.update({
            "super_charge": self._charge,
            "super_ready": False,
            "charging": now < self._charging_until,
        })

    def _reset_tracking(self, preserve_progress: bool = False) -> None:
        """Сбросить устаревшие кадры и жесты; краткий сбой сохраняет прогресс."""
        now = time.monotonic()
        self.frame = None
        if not preserve_progress:
            self._calib_samples = None
            self._charge = 0.0
            self._last_blink_time = None
            self._charge_updated_at = now
            self._charging_until = 0.0
        else:
            self._update_charge(now)
        self._prev_fist = False
        self._fist_held = False
        self._prev_blink = False
        self._fps_window_start = time.monotonic()
        self._fps_frames = 0
        self.gaze_flt_x = OneEuro(min_cutoff=1.2, beta=0.3)
        self.gaze_flt_y = OneEuro(min_cutoff=1.2, beta=0.3)
        self.hand_flt_x = OneEuro(min_cutoff=1.0, beta=0.2)
        self.hand_flt_y = OneEuro(min_cutoff=1.0, beta=0.2)
        self.smile_ema = EMA(CFG.smoothing_alpha)
        self.mouth_ema = EMA(CFG.smoothing_alpha)
        self.state.update({
            "camera": False, "camera_open": False, "face": False,
            "hand_present": False, "fist": False, "fps": 0.0,
            "aim_x": FIELD_CENTER, "aim_y": FIELD_CENTER,
            "hand_x": FIELD_CENTER, "hand_y": FIELD_CENTER,
            "iris_x": 0.0, "iris_y": 0.0, "gaze_dx": 0.0, "gaze_dy": 0.0,
            "smile_score": 0.0, "mouth_open_score": 0.0,
            "smile": False, "mouth_open": False,
            "calibrating": self._calib_samples is not None,
            "super_charge": self._charge, "super_ready": False,
            "blink_count": 0, "charging": now < self._charging_until,
        })

    def start_calibration(self) -> None:
        """Запомнить нейтраль взгляда: усредняем следующие CALIB_FRAMES кадров."""
        with self.lock:
            self._calib_samples = []
            self.state["calibrating"] = True

    @staticmethod
    def _robust_center(values: list[float]) -> float:
        """Медиана с отбрасыванием выбросов (3σ по MAD).

        Раньше нейтраль считалась средним арифметическим: одно моргание или
        рывок зрачка утаскивали её вбок. Медиана с отсечкой выбросов держит
        нейтраль на реальном центре взгляда.
        """
        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])
        if mad == 0:
            return med
        tol = 3.0 * 1.4826 * mad
        keep = [v for v in values if abs(v - med) <= tol]
        return sum(keep) / len(keep) if keep else med

    def _collect_calibration(self, gaze_x: float, gaze_y: float,
                             blink: bool = False) -> None:
        if self._calib_samples is None:
            return
        if blink:
            # Моргнул — зрачок на кадре пропал, такой замер только портит нейтраль.
            return
        self._calib_samples.append((gaze_x, gaze_y))
        if len(self._calib_samples) < self.CALIB_FRAMES:
            return
        xs = [sample[0] for sample in self._calib_samples]
        ys = [sample[1] for sample in self._calib_samples]
        CFG.gaze_neutral_x = self._robust_center(xs)
        CFG.gaze_neutral_y = self._robust_center(ys)
        self._calib_samples = None
        CFG.save()

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

    HAND_PALM = (0, 5, 9, 13, 17)   # запястье + основания пальцев = центр ладони
    HAND_TIPS = (8, 12, 16, 20)     # кончики пальцев

    def _detect_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def _update_hand(self, hand_lm) -> None:
        if hand_lm is None:
            self.state["hand_present"] = False
            self.state["fist"] = False
            self._fist_held = False
            self.hand_flt_x = OneEuro(min_cutoff=1.0, beta=0.2)
            self.hand_flt_y = OneEuro(min_cutoff=1.0, beta=0.2)
            return
        lm = hand_lm.landmark
        cx = sum(lm[i].x for i in self.HAND_PALM) / len(self.HAND_PALM)
        cy = sum(lm[i].y for i in self.HAND_PALM) / len(self.HAND_PALM)

        # Раскрытие ладони: как далеко кончики пальцев от центра ладони,
        # в единицах размера кисти. Открытая ладонь — далеко, кулак — близко.
        d = sum(math.hypot(lm[i].x - cx, lm[i].y - cy) for i in self.HAND_TIPS) \
            / len(self.HAND_TIPS)
        scale = math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) + 1e-6
        openness = d / scale

        # Гистерезис кулака: не дёргается на границе порога.
        if self._fist_held:
            if openness > CFG.fist_release:
                self._fist_held = False
        elif openness < CFG.fist_threshold:
            self._fist_held = True

        now = time.monotonic()
        hx = self.hand_flt_x.update(cx, now)
        hy = self.hand_flt_y.update(cy, now)
        self.state["hand_present"] = True
        self.state["hand_x"] = hx
        self.state["hand_y"] = hy
        self.state["fist"] = self._fist_held

    def _aim_from_hand(self):
        """Рука двигает прицел напрямую: положение ладони -> доля поля.

        По X зеркалим: в сыром кадре правая рука едет влево по изображению,
        а игрок ждёт «рука вправо = прицел вправо».
        """
        hx = 1.0 - self.state["hand_x"]
        hy = self.state["hand_y"]
        ax = clamp01(FIELD_CENTER + (hx - FIELD_CENTER) * CFG.hand_gain)
        ay = clamp01(FIELD_CENTER + (hy - FIELD_CENTER) * CFG.hand_gain)
        return ax, ay

    def _open_capture(self) -> None:
        """Открыть камеру: сначала DSHOW, потом бэкенд по умолчанию."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(CFG.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(CFG.camera_index)

    def start(self):
        with self.lifecycle_lock:
            if self._closed:
                raise RuntimeError("Трекер камеры уже закрыт")
            if self.thread is not None and self.thread.is_alive():
                if self.running:
                    return
                raise RuntimeError("Предыдущий поток камеры ещё останавливается")
            with self.lock:
                self._reset_tracking()
                self.state["error"] = None
            try:
                self._open_capture()
                if not self.cap.isOpened():
                    raise RuntimeError("Не удалось открыть камеру")
            except Exception as exc:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                with self.lock:
                    self.state["error"] = str(exc)
                raise
            with self.lock:
                self._stop_event.clear()
                self.state["camera_open"] = True
                self.running = True
                self.thread = threading.Thread(target=self._loop, daemon=True)
            try:
                self.thread.start()
            except Exception as exc:
                self.cap.release()
                self.cap = None
                self.thread = None
                with self.lock:
                    self.running = False
                    self._reset_tracking()
                    self.state["error"] = str(exc)
                raise

    def stop(self):
        with self.lifecycle_lock:
            with self.lock:
                self.running = False
                self._stop_event.set()
                self._reset_tracking()
                self.state["error"] = None
                thread = self.thread
            # Поток сам освобождает камеру: release одновременно с read()
            # может аварийно завершить OpenCV. Лок состояния здесь уже свободен.
            if thread is not None:
                thread.join(timeout=self.STOP_TIMEOUT)
                if thread.is_alive():
                    message = "Камера ещё останавливается: драйвер не завершил чтение кадра"
                    with self.lock:
                        self.state["error"] = message
                    raise RuntimeError(message)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.thread = None

    def close(self):
        """Завершить камеру и графы MediaPipe при выходе из сервера."""
        with self.lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.stop()
            except RuntimeError:
                # Зависший поток — daemon; новый поток стартовать не разрешаем.
                return
            self.face_mesh.close()
            self.hands.close()

    def _loop(self):
        read_failures = 0
        last_reopen = 0.0
        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if self._stop_event.is_set():
                    break
                if not ret:
                    read_failures += 1
                    with self.lock:
                        if read_failures == 1:
                            self._reset_tracking(preserve_progress=True)
                        self.state["camera_open"] = self.cap.isOpened()
                    if read_failures >= self.MAX_READ_FAILURES:
                        now = time.monotonic()
                        if now - last_reopen >= self.REOPEN_COOLDOWN:
                            last_reopen = now
                            self._open_capture()
                            with self.lock:
                                self.state["camera_open"] = self.cap.isOpened()
                        with self.lock:
                            self.state["error"] = (
                                "Камера открыта, но не отдаёт кадры. Устройство занято "
                                "другим процессом (или камера выключена)."
                            )
                    self._stop_event.wait(0.02)
                    continue
                read_failures = 0
                with self.lock:
                    if self._stop_event.is_set():
                        break
                    self.frame = frame
                    self.state["error"] = None
                    self.state["frames_read"] += 1
                    self._fps_frames += 1
                    now = time.monotonic()
                    if now - self._fps_window_start >= 1.0:
                        self.state["fps"] = self._fps_frames / (now - self._fps_window_start)
                        self._fps_window_start = now
                        self._fps_frames = 0
                    try:
                        self._process_frame(frame)
                    except Exception as exc:
                        self._reset_tracking(preserve_progress=True)
                        self.state["camera_open"] = self.cap.isOpened()
                        self.state["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            with self.lock:
                self.state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            with self.lock:
                self.running = False
                self._reset_tracking()

    def _process_frame(self, frame):
        hand_lm = self._detect_hand(frame)
        self._update_hand(hand_lm)

        mesh = self._make_mesh(frame)
        self.state["face"] = bool(mesh)

        now = time.monotonic()
        self._update_charge(now)

        # Обычный выстрел — смыкание руки в кулак.
        shots = 0
        fist_now = self.state["hand_present"] and self.state["fist"]
        if fist_now and not self._prev_fist:
            shots = 1
        self._prev_fist = fist_now

        if not mesh:
            # Лица нет: заряжать супер нечем, но рукой целиться и стрелять можно.
            if self.state["hand_present"]:
                ax, ay = self._aim_from_hand()
            else:
                ax, ay = self.state["aim_x"], self.state["aim_y"]
            # Потеря лица не считается завершением моргания при его возвращении.
            self._prev_blink = False
            self.gaze_flt_x = OneEuro(min_cutoff=1.2, beta=0.3)
            self.gaze_flt_y = OneEuro(min_cutoff=1.2, beta=0.3)
            self.state.update({
                "camera": True,
                "aim_x": ax,
                "aim_y": ay,
                "smile_score": 0.0,
                "mouth_open_score": 0.0,
                "iris_x": 0.0,
                "iris_y": 0.0,
                "gaze_dx": 0.0,
                "gaze_dy": 0.0,
                "calibrating": self._calib_samples is not None,
                "shot_id": self.state["shot_id"] + shots,
            })
            return

        lm = mesh.landmark

        # Взгляд: зрачок относительно центра своего глаза, в размерах глаза
        gaze_lx, gaze_ly, eye_lw = eye_gaze(lm[LEFT_IRIS_CENTER].x, lm[LEFT_IRIS_CENTER].y,
                                            lm[LEFT_EYE_OUTER].x, lm[LEFT_EYE_INNER].x,
                                            lm[LEFT_EYE_TOP].y, lm[LEFT_EYE_BOTTOM].y)
        gaze_rx, gaze_ry, eye_rw = eye_gaze(lm[RIGHT_IRIS_CENTER].x, lm[RIGHT_IRIS_CENTER].y,
                                            lm[RIGHT_EYE_OUTER].x, lm[RIGHT_EYE_INNER].x,
                                            lm[RIGHT_EYE_TOP].y, lm[RIGHT_EYE_BOTTOM].y)
        raw_gx = (gaze_lx + gaze_rx) / 2.0
        raw_gy = (gaze_ly + gaze_ry) / 2.0
        gaze_x = self.gaze_flt_x.update(raw_gx, now)
        gaze_y = self.gaze_flt_y.update(raw_gy, now)
        eye_w = (eye_lw + eye_rw) / 2.0     # масштаб лица для метрик рта

        # Моргание: раскрытие глаза (высота/ширина) ниже порога = глаз закрыт.
        left_open = abs(lm[LEFT_EYE_TOP].y - lm[LEFT_EYE_BOTTOM].y)
        right_open = abs(lm[RIGHT_EYE_TOP].y - lm[RIGHT_EYE_BOTTOM].y)
        eye_open = (left_open + right_open) / 2.0
        blink = (eye_open / eye_w) < self.BLINK_RATIO

        # Закрылся -> открылся = одна порция энергии; удержание глаз не заряжает.
        completed_blink = not blink and self._prev_blink
        if completed_blink:
            self._charge = min(1.0, self._charge + CFG.super_blink_gain)
            self._last_blink_time = now
            self._charge_updated_at = now
            self._charging_until = now + self.CHARGE_PULSE_SECONDS
            self.state["blink_id"] += 1
        self._prev_blink = blink

        # Полная сумма энергии автоматически выпускает супер-выстрел.
        super_shot = 0
        if completed_blink and (self._charge >= 1.0 or math.isclose(self._charge, 1.0, abs_tol=1e-12, rel_tol=0.0)):
            super_shot = 1
            self._charge = 0.0

        # Улыбка и открытый рот — только как диагностика (не стреляют).
        mouth_w = abs(lm[MOUTH_LEFT].x - lm[MOUTH_RIGHT].x)
        smile_score = self.smile_ema.update(mouth_w / eye_w)
        mouth_h = abs(lm[LIP_TOP_INNER].y - lm[LIP_BOTTOM_INNER].y)
        mouth_open_score = self.mouth_ema.update(mouth_h / eye_w)

        self._collect_calibration(raw_gx, raw_gy, blink)

        # Прицел — только рука.
        if self.state["hand_present"]:
            aim_x, aim_y = self._aim_from_hand()
        else:
            aim_x, aim_y = self.state["aim_x"], self.state["aim_y"]

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
            "smile": False,
            "mouth_open": False,
            "fist": fist_now,
            "super_charge": self._charge,
            "super_ready": False,
            "blink_count": 0,
            "charging": now < self._charging_until,
            "calibrating": self._calib_samples is not None,
            "shot_id": self.state["shot_id"] + shots,
            "super_id": self.state["super_id"] + super_shot,
        })

    def get_frame(self):
        with self.lock:
            return self.frame

    def get_state(self):
        with self.lock:
            self._update_charge(time.monotonic())
            # Копия, а не ссылка: jsonify сериализует словарь уже без лока,
            # а поток трекинга в это время может менять self.state.
            return dict(self.state)

# Инициализация трекера
tracker = CameraTracker()
atexit.register(tracker.close)

# Создание Flask приложения
app = Flask(__name__)

# MJPEG поток
def mjpeg_stream():
    while True:
        frame = tracker.get_frame()
        if frame is None:
            if not tracker.running:
                return
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
        time.sleep(1.0 / 30.0)

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
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Ожидается JSON-объект настроек"}), 400
        bounds = {
            "level_duration": (10, 600, int),
            "level_difficulty": (1, 5, int),
            "super_blink_count": (2, 8, int),
            "super_blink_gain": (0.05, 1.0, float),
            "super_decay_delay": (0.5, 15.0, float),
            "super_decay_rate": (0.0, 0.5, float),
        }
        for key, (minimum, maximum, number_type) in bounds.items():
            if key not in data:
                continue
            try:
                value = data[key]
                converted = number_type(value)
                if isinstance(value, bool) or not math.isfinite(converted) or not minimum <= converted <= maximum:
                    raise ValueError
                if number_type is int and float(value) != converted:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                kind = "целое число" if number_type is int else "число"
                return jsonify({"error": f"{key}: {kind} от {minimum} до {maximum}"}), 400
        with tracker.lock:
            # Учитываем время по старой скорости затухания перед сменой настройки.
            tracker._update_charge(time.monotonic())
            CFG.update(**data)
            CFG.save()
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
    with tracker.lock:
        tracker._calib_samples = None
        tracker.state["calibrating"] = False
        CFG.gaze_neutral_x = 0.0
        CFG.gaze_neutral_y = 0.0
        tracker.state["neutral_x"] = 0.0
        tracker.state["neutral_y"] = 0.0
        CFG.save()
    return jsonify({"ok": True})

# Сохранение рекорда: принимает счёт, запоминает лучший и пишет в config.json
@app.route('/score', methods=['POST'])
def score():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Ожидается JSON-объект со счётом"}), 400
    try:
        s = int(data.get("score", 0))
    except (TypeError, ValueError, OverflowError):
        s = 0
    with tracker.lock:
        if s > CFG.best_score:
            CFG.best_score = s
            CFG.save()
        return jsonify({"best_score": CFG.best_score})

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
    try:
        tracker.stop()
        return jsonify({"ok": True})
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503

# Запуск приложения
if __name__ == "__main__":
    suppress_noise()
    if os.environ.get("SMILEGUN_SKIP_CAMERA") != "1":
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
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        tracker.close()
