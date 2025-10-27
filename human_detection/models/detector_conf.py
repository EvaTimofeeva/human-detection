"""
Конфигурация детектора (общие настройки для всех моделей).
Вынесено в отдельный файл, чтобы не плодить циклические импорты.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectorConfig:
    """
    контейнер с настройками детектора.
    """

    model: str = "yolov8"  # "yolov8", "fasterrcnn", "hog", "auto"
    model_path: str = "yolov8n.pt"  # для YOLOv8
    conf: float = 0.35  # минимальный score для детекции
    iou: float = 0.45  # порог IoU в NMS (чтобы боксы не пересекались)
    imgsz: int = 640
    device: str = "cpu"
    half: bool = True
    classes: Optional[List[int]] = field(
        default_factory=lambda: [0]
    )  # оставляем только класс людей
