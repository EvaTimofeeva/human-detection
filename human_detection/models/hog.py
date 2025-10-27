"""
OpenCV HOG+SVM —  детектор пешеходов
Никаких внешних весов не требуется — используется встроённый SVM детектор пешеходов OpenCV
Возвращает боксы людей.
Измеряем "уверенность" по весам (weights) и нормализуем через сигмоиду, чтобы получить числа в промежутке [0,1]
"""

from typing import Tuple
import numpy as np
import cv2

from human_detection.models.base_detector import BaseDetector
from human_detection.models.detector_conf import DetectorConfig


class PeopleDetectorHOG(BaseDetector):
    """
    Детектор пешеходов HOGDescriptor + SVM из OpenCV.
    Возвращает боксы людей. Уверенность оценим через расстояние до границы.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.class_names = ["person"]

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # detectMultiScale возвращает прямоугольники (x, y, w, h) и веса (weights)
        rects, weights = self.hog.detectMultiScale(
            frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        if len(rects) == 0:  # если пусто — возвращаем пустые массивы
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        # переводим (x, y, w, h) в [x1, y1, x2, y2]
        boxes = np.zeros((len(rects), 4), dtype=np.float32)
        for i, (x, y, w, h) in enumerate(rects):
            boxes[i] = [x, y, x + w, y + h]

        # weights будем понимать как "уверенность"; нормализуем к [0,1] грубо, через сигмоиду
        w_arr = np.array(weights).reshape(-1).astype(np.float32)
        conf_scores = 1 / (1 + np.exp(-w_arr))  # очень приблизительно нормализуем

        # Защита на случай рассинхрона (очень редко, но по аналогии)
        n = min(len(boxes), len(conf_scores))
        boxes = boxes[:n]
        conf_scores = conf_scores[:n]

        # Порог по уверенности
        mask = conf_scores >= float(self.cfg.conf)
        boxes = boxes[mask]
        conf_scores = conf_scores[mask]

        cls = np.zeros((boxes.shape[0],), dtype=np.int32)
        return boxes, conf_scores, cls
