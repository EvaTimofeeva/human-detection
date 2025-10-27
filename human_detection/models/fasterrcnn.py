"""
Faster R-CNN (torchvision) — детектор людей
Используем предобученные веса COCO (DEFAULT). Подтянутся автоматически при первом запуске и попадут в кэш Torch.
Тут класс 'person' имеет label=1, возвращаем class_ids=0 для единообразия.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

from human_detection.models.base_detector import BaseDetector
from human_detection.models.detector_conf import DetectorConfig

import torch
import torchvision
from torchvision.transforms import functional as F


class PeopleDetectorFRCNN(BaseDetector):
    """Детектор на базе torchvision.models.detection.fasterrcnn_resnet50_fpn."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.device = (
            torch.device("cuda")
            if (str(cfg.device) != "cpu" and torch.cuda.is_available())
            else torch.device("cpu")
        )

        # загружаем предобученную модель
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        self.model.eval().to(self.device)

        self.person_class_index = 1
        self.class_names = ["person"]

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Преобразуем BGR в RGB и в тензор
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])[0]

        boxes_t = outputs.get("boxes", None)
        scores_t = outputs.get("scores", None)
        labels_t = outputs.get("labels", None)

        if (
            boxes_t is None
            or scores_t is None
            or labels_t is None
            or boxes_t.shape[0] == 0
        ):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        boxes = boxes_t.detach().cpu().numpy().astype(np.float32)
        conf_scores = scores_t.detach().cpu().numpy().astype(np.float32)
        labels = labels_t.detach().cpu().numpy().astype(np.int32)

        # На всякий: усечём до общего минимума
        n = min(len(boxes), len(conf_scores), len(labels))
        boxes = boxes[:n]
        conf_scores = conf_scores[:n]
        labels = labels[:n]

        # оставляем только person + порог уверенности
        mask = (labels == self.person_class_index) & (
            conf_scores >= float(self.cfg.conf)
        )
        boxes = boxes[mask]
        conf_scores = conf_scores[mask]

        cls = np.zeros((boxes.shape[0],), dtype=np.int32)  # один класс: person
        return boxes, conf_scores, cls
