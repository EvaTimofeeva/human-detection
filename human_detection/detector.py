"""
Детектор (Ultralytics YOLOv8)
Загружаем модель и вызываем predict на каждом кадре.
"""

from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    conf: float = 0.3  # минимальный score для детекции
    iou: float = 0.45  # порог IoU в NMS (чтобы боксы не пересекались)
    imgsz: int = 640
    device: str = "cpu"
    half: bool = True
    classes: Optional[List[int]] = field(
        default_factory=lambda: [0]
    )  # оставляем только класс людей


# обьявляем класс, в который завернем модельку YOLOv8
class PeopleDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        """Инициализация детектора: загрузка модели и сохранение конфигурации"""
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.class_names: List[str] = self.model.names

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Выполняет детекцию объектов на одном кадре.
        :param frame_bgr: Входной кадр (BGR numpy array)
        :return: Кортеж из трех массивов NumPy: (boxes, conf_scores, class_ids)
        """
        result = self.model.predict(
            source=frame_bgr,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        # Результаты модели хранятся в объекте result.boxes в формате Tensors

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)

        # [x_min, y_min, x_max, y_max] - координаты рамок, получаем по атрибуту .xyxy
        conf_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        cls = result.boxes.cls.detach().cpu().numpy().astype(np.int32)  # class ID

        if self.cfg.classes is not None:
            mask = np.isin(cls, np.array(self.cfg.classes, dtype=np.int32))
            boxes, conf_scores, cls = boxes[mask], conf_scores[mask], cls[mask]

        return boxes, conf_scores, cls
