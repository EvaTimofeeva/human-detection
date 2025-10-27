"""
Конфигурация детектора и создание конкретной модели.
"""

from typing import Tuple

import numpy as np

from human_detection.models.yolo import PeopleDetectorYOLO
from human_detection.models.fasterrcnn import PeopleDetectorFRCNN
from human_detection.models.hog import PeopleDetectorHOG
from human_detection.models.base_detector import BaseDetector
from human_detection.models.detector_conf import DetectorConfig


def create_detector(cfg: DetectorConfig) -> BaseDetector:
    """Создаем выбранный детектор"""
    model = (cfg.model or "yolov8").lower()
    if model == "yolov8":
        return PeopleDetectorYOLO(cfg)
    if model == "fasterrcnn":
        return PeopleDetectorFRCNN(cfg)
    if model == "hog":
        return PeopleDetectorHOG(cfg)
    raise ValueError(f"model, которую мы не знаем: {cfg.model}")


class PeopleDetector(BaseDetector):
    """Оболочка для любого детектора"""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.inner = create_detector(cfg)
        self.class_names = self.inner.class_names

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.inner.infer(frame_bgr)
