# SmileGun

컴퓨터 비전과 손/얼굴 동작으로 제어하는 실험적 브라우저 게임입니다. 웹캠이 컨트롤러가 되어 손 움직임으로 조준하고, 제스처로 발사하며, 얼굴 동작으로 힘을 충전합니다.

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
