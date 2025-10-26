"""
Рисует рамки и подписи
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class VizConfig:
    box_thickness: int = 10
    font_scale: float = 0.7
    alpha: float = 0.06
    palette_seed: int = 42


def _color_person(seed: int = 42) -> Tuple[int, int, int]:
    """
    Возвращает цвет (BGR) для класса 'person'.
    """
    return (10, 255, 10)


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    class_names: List[str],
    cfg: VizConfig,
) -> np.ndarray:
    """Рисуем прямоугольники и подписи (класс + confidences)
    args:
    frame: входной BGR-кадр
    boxes_xyxy: массив Nx4 с координатами [x1, y1, x2, y2]
    confidences: массив Nx1 [0..1]
    class_ids: не используется (совместимость)
    class_names: не используется (совместимость)
    cfg: параметры визуализации
    returns:
    Кадр с отрисованными рамками и подписями.

    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return frame

    # Создаем копию кадра, которую будем использовать для отрисовки полупрозрачного фона
    overlay = frame.copy()
    h, w = frame.shape[:2]
    color = _color_person(cfg.palette_seed)

    # идем циклом по каждой обнаруженной детекции
    for i in range(len(boxes_xyxy)):
        # координаты
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        x1 = max(0, min(x1, w - 1))  # приводим координаты к границам кадра
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # рамка
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.box_thickness)

        # подпись
        conf = float(confidences[i]) if len(confidences) > i else 0.0
        label = f"person {conf:.2f}"

        # Вычисление размеров текста и позиции
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, 1
        )
        th_total = th + baseline + 4  # Общая высота подложки текста с отступами
        y_top = max(0, y1 - th_total)  # Верхняя координата подложки

        # фон для подписи
        cv2.rectangle(overlay, (x1, y_top), (x1 + tw + 6, y_top + th_total), color, -1)
        cv2.addWeighted(
            overlay, cfg.alpha, frame, 1 - cfg.alpha, 0, frame
        )  # Используем cfg.alpha для прозрачности

        # текст
        cv2.putText(
            frame,
            label,
            (x1 + 3, y_top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.font_scale,
            (255, 255, 255),  # белый
            1,
            cv2.LINE_AA,
        )

    return frame
