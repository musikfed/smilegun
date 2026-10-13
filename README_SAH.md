# SmileGun

Компьютернай көрүү, илии уонна сирэй хамсаныыттан саланар экспериментальнай browser оонньуу. Webcam контроллер буолар: илии быһаарар, жест атыылыыр, сирэй хамсаныыта күүһү хомуйар.

## Architecture / Архитектура

```text
Browser webcam/game UI -> Flask endpoints -> MediaPipe/OpenCV analysis -> aim/gesture/blink events -> browser game state and generated audio
```

## Project structure / Структура проекта

- Python/Flask backend — camera and game API
- MediaPipe/OpenCV — hand and face analysis
- browser UI — game rendering and controls
- PowerShell launcher — local Windows startup

## Contact / Контакты

- Email: feudor.lab@yandex.ru
- Telegram: @Feudorlab1
