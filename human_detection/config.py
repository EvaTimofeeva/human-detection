"""
Cбор конфигураций для pipeline.
Парсинг аргументов командной строки.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .visualize import VizConfig
from human_detection.detector import DetectorConfig


root = Path(__file__).resolve().parents[1]
input_data_path = root / "input_data" / "crowd.mp4"
output_data_path = root / "output_data" / "annotated.mp4"
metrics_dir_path = root / "human_detection" / "metrics"


@dataclass  # декоратор из модуля dataclasses, который автоматически создает методы для класса по его аннотациям типов
class AppConfig:
    """Конфигурация приложения, которую принимает pipeline"""

    input_path: Path
    output_path: Path
    metrics_dir: Path
    save_frames: bool
    save_every: int
    show: bool
    fps_override: Optional[float]
    max_frames: Optional[int]
    detector: DetectorConfig
    viz: VizConfig


def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки"""
    # создаём парсер аргументов
    p = argparse.ArgumentParser(description="Детекция людей на видео")

    # добавляем аргументы
    p.add_argument(
        "--input", default=str(input_data_path), help="Путь к входному видео."
    )
    p.add_argument(
        "--output", default=str(output_data_path), help="Путь к выходному видео."
    )
    p.add_argument(
        "--metrics-dir",
        default=str(metrics_dir_path),
        help="Папка для CSV/JSON метрик.",
    )

    # добавляем аргументы для детектора
    p.add_argument("--model", default="yolov8n.pt", help="Весa YOLO (имя или путь).")
    p.add_argument("--conf", type=float, default=0.35, help="Порог уверенности.")
    p.add_argument("--iou", type=float, default=0.45, help="Порог IoU для NMS.")
    p.add_argument("--imgsz", type=int, default=640, help="Размер входа модели (px).")
    p.add_argument(
        "--device", default="cpu", help='Устройство: "cpu" или индекс GPU, напр. "0".'
    )
    p.add_argument(
        "--half", action="store_true", help="Включить FP16 (если поддерживается)."
    )
    p.add_argument(
        "--no-half", action="store_true", help="Принудительно отключить FP16."
    )
    p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=[0],
        help="ID классов (COCO: 0=person).",
    )

    # добавляем аргументы для визуализации
    p.add_argument("--show", action="store_true", help="Показывать окно предпросмотра.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Обработать только первые N кадров (0 = все).",
    )
    p.add_argument(
        "--imgsave", action="store_true", help="Сохранять выборочные кадры с разметкой."
    )
    p.add_argument(
        "--img-every", type=int, default=50, help="Каждый N-й кадр при --imgsave."
    )
    p.add_argument(
        "--fps-override", type=float, default=None, help="Принудительный FPS на выходе."
    )

    # парсим аргументы и возвращаем их в виде объекта Namespace
    return p.parse_args()


def make_configs(ns: argparse.Namespace) -> AppConfig:
    """Создаёт конфигурационные объекты для pipeline"""
    # оборачиваем в датасеты для удобства

    max_frames = ns.max_frames if ns.max_frames and ns.max_frames > 0 else None

    # создаём конфигурацию для детектора
    det_cfg = DetectorConfig(
        model_path=ns.model,
        conf=ns.conf,
        iou=ns.iou,
        imgsz=ns.imgsz,
        device=ns.device,
        half=(ns.half and not ns.no_half),
        classes=[0],
    )

    # создаём конфигурацию для визуализации
    viz_cfg = VizConfig(
        box_thickness=2,  # толщина границы прямоугольника в пикселях
        font_scale=0.5,  # размер шрифта
        alpha=0.25,  # прозрачность
        palette_seed=42,  # для палитры цветов
    )

    return AppConfig(
        input_path=Path(ns.input),
        output_path=Path(ns.output),
        metrics_dir=Path(ns.metrics_dir),
        save_frames=bool(ns.imgsave),
        save_every=int(ns.img_every),
        show=bool(ns.show),
        fps_override=ns.fps_override,
        max_frames=max_frames,
        detector=det_cfg,
        viz=viz_cfg,
    )
