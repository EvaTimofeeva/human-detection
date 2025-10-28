# Детекция людей на видео с отрисовкой боксов, подписи класса, confident.

Точка входа - main.py

Поддерживаются 3 модели детекции людей:

1) **yolov8** (Ultralytics, веса .pt)
2) **fasterrcnn** (torchvision, COCO)
3) **hog** (классический OpenCV HOG+SVM)

Есть режим сравнения моделей: по видео пробегаются все модели, собираются метрики и выбирается лучшая.

### Структура проекта:
```
human_detection/
├─ main.py                  # точка входа
├─ config.py                # аргументы командной строки, сбор AppConfig
├─ pipeline.py              # чтение видео → инференс → отрисовка → запись → отчёты
├─ video.py                 # безопасное чтение/запись видео (OpenCV)
├─ visualize.py             # отрисовка боксов и подписей
├─ utils
│  ├─ utils.py              # вспомогательные функции
├─ models/
│  ├─ base_detector.py      # BaseDetector интерфейс
│  ├─ detector_conf.py      # DetectorConfig
│  ├─ yolo.py               # YOLOv8 (Ultralytics)
│  ├─ fasterrcnn.py         # Faster R-CNN (torchvision)
│  └─ hog.py                # HOG+SVM (OpenCV)
└─ metrics/                
   └─  report.py           # сбор метрик по кадрам          
```

- также используется [pre-commit](https://pre-commit.com) (см. `.pre-commit-config.yaml`) для автоматического линтера и фиксации зависимостей

## Установка и запуск:
### 1.1) Установка и запуск через пакетный менеджер uv:
Установка uv: 

`pip install uv`

Создание виртуального окружения и установка зависимостей: 

`uv sync`

Базовый запуск приложения (YOLOv8):

`uv run main.py --input crowd.mp4`

Режим сравнения 3 моделей:

`uv run main.py --compare --input crowd.mp4`

для запуска на GPU:

`uv run main.py --input crowd.mp4 --device 0`


### 1.2) Установка и запуск через pip:
python v.3.12

`pip install -r requirements.txt`

`python main.py --compare --input crowd.mp4`

### Просмотр доступных параметров:
`uv run main.py --help`
```
options:
  -h, --help            show this help message and exit
  --input INPUT         Путь к входному видео.
  --output OUTPUT       Путь к выходному видео.
  --metrics-dir METRICS_DIR
                        Папка для CSV/JSON метрик.
  --backend {yolov8,fasterrcnn,hog,auto}
                        Выбор бэкенда: "yolov8", "fasterrcnn", "hog", либо "auto" для сравнения.
  --model MODEL         Весa YOLO.
  --conf CONF           Порог уверенности.
  --iou IOU             Порог IoU для NMS.
  --imgsz IMGSZ         Размер входа модели (px) для YOLOv8.
  --device DEVICE       Устройство: "cpu" или индекс GPU, напр. "0".
  --half                Включить FP16 (если поддерживается).
  --no-half             Принудительно отключить FP16.
  --compare             Сравним несколько моделей (yolov8, fasterrcnn, hog) и выберем лучшую.
  --compare-backends [COMPARE_BACKENDS ...]
                        Какие бэкенды сравнивать (по умолчанию: yolov8 fasterrcnn hog).
  --show                Показывать окно предпросмотра.
  --max-frames MAX_FRAMES
                        Обработать только первые N кадров (0 = все).
  --imgsave             Сохранять выборочные кадры с разметкой.
  --img-every IMG_EVERY
                        Каждый N-й кадр при --imgsave.
  --fps-override FPS_OVERRIDE
                        Принудительный FPS на выходе.
```
## Что получится:

- На каждую модель — своё видео в папке output_data/

- Таблица сравнения: metrics/models_comparison.csv

## Сравнение моделей и метрики:

Так как у нас нет разметки, считаем две метрики:

**persons_per_frame** — людей на кадр в среднем (прокси на полноту/recall)

**fps_mean** — средний FPS (скорость на железе)

### Финальный conf_score:
`conf_score = persons_per_frame * 1.0 + fps_mean * 0.2`

(т.е. качество важнее скорости, но скорость тоже учитываем)

## Итог:
<table> <tr> <td align="center">YOLOv8</td> <td align="center">Faster R-CNN</td> <td align="center">HOG</td> </tr> <tr> <td><img src="./assets/yolov8.gif" alt="YOLOv8 demo" /></td> <td><img src="./assets/frrcnn.gif" alt="Faster R-CNN demo" /></td> <td><img src="./assets/hog.gif" alt="HOG demo" /></td> </tr> </table>


| Бэкенд       | Кадров | Детекций | Людей | FPS (ср.) | Runtime (с) |score |
|--------------|:------:|---------:|------:|----------:|------------:|-----:|     
| YOLOv8       | 705    | 7242     | 7242  | 118.07    |  16.75      | 33.8 |
| Faster R-CNN | 705    | 29302    | 29302 | 6.77      | 120.13      | 42.9 |
| HOG          | 705    | 8032     | 8032  | 1.01      | 709.67      | 11.6 |
                        
- **YOLOv8** даёт наименьшее число обнаруженных людей, но быстрая скорость.
- **Faster R-CNN** очень медленный, но лучше всех детектирует людей.
- **HOG** —  очень медленный, но по метрикам лучше yolo, но на итоговом видео заметно много пропусков на сложных сценах и много ошибочных боксов.

Лучшая модель для этого видео: **Faster R-CNN**

## шаги по улучшению :

Чтобы предпринимать шаги по улучшению необходимо определиться с метриками по которым мы будем отслеживать прогресс работы программы.

1) Уменьшить входной размер кадров

2) Ограничить количество боксов после NMS

3) Попробовать CUDNN autotune (если меняется размер кадров): `torch.backends.cudnn.benchmark = True`

4) Подобрать `--conf/--iou` для YOLOv8 (возможно, чуть повысить `conf`, чтобы снизить ложные срабатывания, и аккуратно подстроить `iou`).

5) Попробовать более крупные веса (yolov8s.pt, yolov8m.pt) — они обычно дают лучший recall на людях при умеренном падении FPS.

6) Тайлинг (tiling)- поделить кадр на 2–4 части и детектировать по отдельности: растёт детальность для дальних людей. Но скорость будет медленнее.

7) Попробовать другие модели

8) Подобрать тестовый набор данных, наиболее подходящий целевой задачи

9) Провести оптимизацию гиперпараметров при помощи, например, [optuna](https://optuna.org)

10) Triton Inference Server (вынести инференс из процесса в сервис)

11) Ускорить Inference через Triton и TensorRT
