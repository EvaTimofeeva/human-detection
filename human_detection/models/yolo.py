"""
YOLOv8 (Ultralytics) — детектор людей.
Возвращает боксы, уверенности и идентификаторы классов
"""

from typing import Tuple
import numpy as np
from ultralytics import YOLO

from human_detection.models.base_detector import BaseDetector
from human_detection.models.detector_conf import DetectorConfig


# обьявляем класс, в который завернем модельку YOLOv8
class PeopleDetectorYOLO(BaseDetector):
    def __init__(self, cfg: DetectorConfig) -> None:
        """Инициализация детектора: загрузка модели и сохранение"""
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

        # Берем имена классов из модели, но оставим интерфейс единым
        try:
            self.class_names = self.model.names
        except Exception:
            self.class_names = ["person"]

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
        # если ничего не нашли — вернём пустые массивы
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

        # Защита от рассинхрона длин массивов
        n = min(len(boxes), len(conf_scores), len(cls))
        if n == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        boxes = boxes[:n]
        conf_scores = conf_scores[:n]
        cls = cls[:n]

        if self.cfg.classes is not None:
            mask = np.isin(cls, np.array(self.cfg.classes, dtype=np.int32))
            # На всякий: если маска пустая — вернём корректные пустые массивы
            if mask.size != n:
                mask = np.zeros((n,), dtype=bool)
            boxes = boxes[mask]
            conf_scores = conf_scores[mask]
            cls = cls[mask]

        if boxes.shape[0] > 0:
            cls[:] = 0  # Приводим class_ids к 0 (person)

        return boxes, conf_scores, cls
