# SmileGun

कंप्यूटर विज़न, हाथ और चेहरे की क्रियाओं से नियंत्रित प्रयोगात्मक ब्राउज़र गेम। वेबकैम कंट्रोलर बनता है: हाथ निशाना साधता है, इशारे गोली चलाते हैं और चेहरे की क्रियाएँ शक्ति चार्ज करती हैं।

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
