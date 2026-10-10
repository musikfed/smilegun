# SmileGun

## English overview

**SmileGun** is an experimental browser game controlled by computer vision and voice-like actions. It explores an alternative game input model where a webcam becomes the controller: hand motion aims, gestures trigger shots, facial actions build power, and the browser UI reacts in real time.

The project is also a practical playground for **human-computer interaction, MediaPipe/OpenCV vision pipelines, low-latency browser ↔ Python communication, generated sound effects, and multimodal controls**.

### Highlights

- hand tracking for aiming;
- gesture-based shooting;
- blink / face actions for charging power;
- configurable levels and difficulty;
- locally generated game sounds;
- Python 3.12 + Flask + MediaPipe + OpenCV;
- designed to run on ordinary consumer hardware.

> Current development is moving toward a browser-camera architecture so the webcam is opened by the browser and Python only receives frames for analysis. See the `versions/v2.1-browser-square` branch for the newer experimental line.

## Author

Built by **musikfed** as an independent experimental project.

- Email: [feudor.lab@yandex.ru](mailto:feudor.lab@yandex.ru)
- Resume / CV #1: [hh.ru profile](https://hh.ru/resume/71c38b43ff112c93240039ed1f4b534d61334d?hhtmFrom=profile)
- Resume / CV #2: [hh.ru profile](https://hh.ru/resume/b0aff0d0ff112d7b210039ed1f536a71536552?hhtmFrom=profile)
- GitHub: [github.com/musikfed](https://github.com/musikfed)

I am interested in **AI-assisted software development, automation, developer tools, computer vision, system integration, backend/infrastructure, and unconventional human-computer interfaces**.

---

Небольшая веб-игра на Python: веб-камера отслеживает руку (прицел и выстрел кулаком) и моргания глаз (зарядка супер-силы).

## Требования

Windows + `uv` + Python 3.12. На Windows сначала убедись, что есть `ffmpeg` в PATH:

```powershell
winget install Gyan.FFmpeg
```

## Запуск

```powershell
cd smilegun
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Открой `http://127.0.0.1:5000`, выбери уровень и нажми **▶ START** в меню.

В сайдбаре можно выбрать **сложность 1–5**, **время уровня 10–600 секунд**,
**заряд за моргание**, **паузу до утечки** и **скорость утечки**. Настройки сохраняются; время и сложность
применяются к следующему уровню. Кнопка **■ ОСТАНОВИТЬ ИГРУ** или **Esc**
возвращает в меню, в том числе во время отсчёта. Камера останавливается
отдельной кнопкой **ОСТАНОВИТЬ**.

## Как это работает

- Камера: OpenCV, поток MJPEG отдаётся на `/video_feed`.
- **Рука**: MediaPipe Hands. Центр ладони двигает прицел, сжатие в кулак
  (`раскрытие ладони < fist_threshold`) даёт одиночный выстрел, гистерезис
  `fist_release` убирает дрожание на границе.
- **Глаза**: MediaPipe Face Mesh. Каждое завершённое моргание (закрыл глаза,
  затем открыл) заряжает **супер-силу** (`super_charge`, 0..1), при этом кубик
  увеличивается и пульсирует. По умолчанию моргание добавляет **25% заряда**.
  После **3 секунд без морганий** заряд постепенно утекает со скоростью
  **8 процентных пунктов в секунду**. Когда сумма достигает **100%**, супер
  автоматически поражает все мишени, которые уже появились на поле, и
  сбрасывает заряд. Чётность и число морганий не имеют значения: учитывается
  накопленная энергия. Удержание закрытых глаз добавляет заряд только после
  открытия; остановка камеры сбрасывает заряд.
- **Калибровка.** «КАЛИБРОВКА» берёт 25 кадров, отбрасывает моргания и
  выбросы (медиана + MAD) и запоминает нейтраль взгляда для зарядки супера.
  Нейтраль сохраняется в `config.json`; её значения видны в диагностике.
  Зарядка морганиями работает без калибровки взгляда.
- Взгляд и рука сглаживаются фильтром **One Euro** (плавно в покое, отзывчиво
  в движении) — прицел не дёргается.
- Кубик краснеет короткой вспышкой (160 мс) в момент выстрела.
- Прицел приходит на клиент в долях поля (0..1) и рисуется с отступом 22 px
  от рамки, поэтому не уезжает за границу.
- На поле появляются красные **мишени**: наводишь прицел рукой и сжимаешь
  кулак — пуля летит снизу вверх и сбивает мишень. Попадания подряд
  складываются в **комбо**.
- **Уровни и Dendy-меню.** Титульный экран в стиле «Денди»: выбор уровня
  (1–5), рекорд, управление рукой (dwell-клик ~0.9 с). Уровень — сколько
  мишеней сбить за заданное время; сложность меняет цель, размер и скорость
  мишеней. На высоких уровнях мишени дрейфуют. Отсчёт «3-2-1» и таймер
  работают в браузере независимо от частоты кадров камеры.
- **Сохранение.** Настройки и рекорд пишутся в `config.json` в корне проекта
  и переживают перезапуск сервера.
- Интерфейс адаптивный: ширина меню и размеры шрифтов привязаны к окну (`vw`,
  `em`, `clamp`), поле масштабируется без верхнего ограничения, превью
  подрезается на низких окнах. При масштабе страницы 50% меню не мельчает, а
  поле занимает всю свободную область.
- Звуки генерируются локально `generate_sound.py`: отдельные эффекты для
  выбора, старта, остановки, отсчёта, выстрела, попадания, промаха,
  накопления энергии, супера, прохождения уровня, победы, проигрыша,
  калибровки и включения/выключения камеры. Каждый импульс зарядки звучит
  выше по мере накопления энергии. Общая громкость регулируется в сайдбаре.

## Настройки (слайдеры в сайдбаре)

| Слайдер | Параметр | Что делает |
|---|---|---|
| Усиление руки | `hand_gain` | насколько ладонь двигает прицел |
| Заряд за моргание | `super_blink_gain` | сколько энергии добавляет завершённое моргание (5–100%) |
| Пауза до утечки | `super_decay_delay` | сколько секунд заряд сохраняется без морганий (0.5–15) |
| Скорость утечки | `super_decay_rate` | сколько заряда теряется в секунду после паузы (0–50%) |
| Сложность | `level_difficulty` | уровень 1–5: цель, размер и скорость мишеней |
| Длительность уровня | `level_duration` | время следующего уровня в секундах (10–600) |
| Скорость пули | `bullet_speed` | пикс/кадр |
| Громкость звука | — | только в браузере, на сервер не отправляется |
| Ширина/высота поля | `field_width/height` | поле масштабируется под окно |

## HTTP-эндпоинты

- `GET /` — страница игры (`templates/index.html`).
- `GET /state` — `aim_x/aim_y`, `hand_present`, `fist`, `super_charge`,
  `super_ready`, `blink_count`, `blink_id`, `super_id`, `charging`, `shot_id`, а также
  диагностика камеры: `frames_read`, `face`, `fps`, `camera_open`, `error`.
- `GET|POST /config` — чтение и изменение параметров из `config.py`.
- `POST /calibrate`, `POST /calibrate/reset` — калибровка взгляда.
- `POST /score` — сохранить рекорд (принимает `{"score": N}`).
- `POST /camera/start`, `POST /camera/stop` — камера.
- `GET /video_feed` — MJPEG-превью.

Проверка окружения: `python diagnostics.py` (без камеры —
`SMILEGUN_SKIP_CAMERA=1`). Подробные логи Flask: `SMILEGUN_VERBOSE=1`.

Проверки игры без камеры и без изменения пользовательского `config.json`:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_final
node test_frontend.cjs
```

`SMILEGUN_SKIP_CAMERA=1` также отключает автоматическое включение камеры
при запуске сервера. Меню и поле отображаются до её включения.

## Если что-то не работает

Сайдбар показывает диагностику: **FPS камеры**, **Кадров всего**, **Лицо в
кадре**, **Рука в кадре**, а красная строка под статистикой — текст ошибки.

- **«НЕТ КАДРОВ»** или красное «Камера открыта, но не отдаёт кадры» —
  устройство занято другим процессом: закрой Teams/Zoom/«Камера» Windows и
  вкладки, использующие веб-камеру, затем нажми «ОСТАНОВИТЬ» и снова
  «ЗАПУСТИТЬ КАМЕРУ». Проверить, кто держит порт: `netstat -ano | findstr :5000`.
- **«НЕТ РУКИ»** — покажи открытую ладонь в кадр, добавь света.
- **Прицел не двигается** — рука должна быть видна камере (см. «Рука в кадре»).
- **Супер не заряжается** — лицо должно быть видно камере. Полностью закрой
  и снова открой глаза: одно завершённое моргание добавляет часть заряда.
  Проверь индикатор «Лицо в кадре», заряд за моргание и параметры утечки.
- **`MessageFactory ... GetPrototype`** — установлена несовместимая версия
  Protobuf. `run.ps1` устанавливает совместимые версии из `requirements.txt`;
  вручную можно выполнить `uv pip install -r requirements.txt`.
- Приложение запускается **одним** процессом (`use_reloader=False`): с
  релоадером Flask поднимал второй процесс, `tracker.start()` выполнялся в
  обоих, камеру открывали два процесса — и тот, что получил устройство, не
  обслуживал браузер. Снаружи это выглядело как чёрное превью и «НЕТ КАМЕРЫ».
- Из-за этого же после правок кода сервер нужно перезапускать руками: автоперезагрузки больше нет.
