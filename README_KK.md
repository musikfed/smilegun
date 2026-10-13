# SmileGun

Компьютерлік көру, қол қимылы және бет әрекеттерімен басқарылатын эксперименттік браузер ойыны. Веб-камера контроллерге айналады: қол нысаналайды, жест атуға, бет қимылы қуат жинауға қолданылады.

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
