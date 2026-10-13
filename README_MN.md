# SmileGun

Компьютер хараа, гар болон нүүрний үйлдлээр удирддаг туршилтын browser тоглоом. Webcam нь controller болж, гар онилж, gesture буудаж, нүүрний үйлдэл хүч цэнэглэнэ.

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
