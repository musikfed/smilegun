# smilegun/diagnostics.py
from __future__ import annotations
import sys
import os
import logging
import importlib
import traceback

# Переменные TF_CPP_MIN_LOG_LEVEL / GLOG_minloglevel / ABSL_MIN_LOG_LEVEL
# выставляются в app.py до импорта mediapipe — здесь они уже бесполезны,
# потому что TensorFlow Lite читает их один раз при инициализации.

REQUIRED = [
    ("flask", "flask"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
]

OPTIONAL = [
    ("scipy", "scipy"),
    ("lameenc", "lameenc"),
]


def check_environment(verbose: bool = True) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    if sys.version_info < (3, 9):
        errors.append(f"Нужен Python >= 3.9, у тебя {sys.version.split()[0]}")

    for mod, pkg in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception as e:
            errors.append(f"Импорт '{mod}' провалился: {e}. Установи: uv pip install {pkg}")

    for mod, pkg in OPTIONAL:
        try:
            importlib.import_module(mod)
        except Exception:
            warns.append(f"Опциональный модуль '{mod}' отсутствует (uv pip install {pkg})")

    # MediaPipe API
    try:
        import mediapipe as mp
        version = getattr(mp, "__version__", "?")
        if not hasattr(mp, "solutions"):
            errors.append(
                f"mediapipe {version} без mp.solutions. "
                f"Откатись: uv pip install 'mediapipe==0.10.21'"
            )
    except Exception as e:
        errors.append(f"mediapipe сломан: {e}")

    # Камера. Пробу открытия можно отключить (SMILEGUN_SKIP_CAMERA=1), чтобы
    # прогнать математику или тесты там, где камеры заведомо нет.
    if os.environ.get("SMILEGUN_SKIP_CAMERA") == "1":
        warns.append("Проверка камеры пропущена (SMILEGUN_SKIP_CAMERA=1)")
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            ok = cap.isOpened()
            cap.release()
            if not ok:
                errors.append("Камера 0 не открывается. Занята другим приложением?")
        except Exception as e:
            warns.append(f"Проверка камеры: {e}")

    # Звук
    from pathlib import Path
    sound = Path(__file__).parent / "static" / "blaster.mp3"
    if not sound.exists():
        warns.append(f"{sound} не найден — запусти generate_sound.py (иначе выстрел без звука)")

    return errors, warns


def suppress_noise(level: str = "WARNING") -> None:
    """Гасит служебный шум в консоли, оставляя ошибки.

    Главный источник — строка вида
    `127.0.0.1 - - [дата] "GET /state HTTP/1.1" 200 -`:
    браузер опрашивает /state каждый кадр анимации, и такая строка летит
    десятки раз в секунду. Ошибки (5xx, падения, предупреждения) остаются
    видны. Вернуть подробные логи: SMILEGUN_VERBOSE=1 при запуске.
    """
    log_level = logging.DEBUG if os.environ.get("SMILEGUN_VERBOSE") == "1" \
        else getattr(logging, level, logging.WARNING)

    # werkzeug пишет по строке на каждый HTTP-запрос
    for name in ("werkzeug", "werkzeug.serving", "werkzeug._internal"):
        logging.getLogger(name).setLevel(log_level)

    # баннеры "Serving Flask app" / "Debug mode"
    try:
        from flask import cli
        cli.show_server_banner = lambda *a, **k: None
    except Exception:
        pass

    # Свой логгер Flask не трогаем: сообщения уровня ERROR он пропускает
    # через lastResort-обработчик, поэтому ошибки остаются видимыми.


def print_report():
    errors, warns = check_environment()
    lines = ["=" * 60, "  DIAGNOSTICS — smilegun", "=" * 60]
    lines += [f"  [!] {w}" for w in warns]
    lines += [f"  [X] {e}" for e in errors]
    if not errors and not warns:
        lines.append("  [OK] Всё чисто.")
    lines.append("=" * 60)
    if errors:
        lines.append("Запуск прерван из-за критических ошибок.")
    print("\n".join(lines))
    # Без сброса буфера отчёт остаётся невидимым, пока stdout перенаправлен
    # в файл/IDE: Python буферизует его блоками, а не строками.
    sys.stdout.flush()
    if errors:
        sys.exit(1)
    print()


if __name__ == "__main__":
    print_report()