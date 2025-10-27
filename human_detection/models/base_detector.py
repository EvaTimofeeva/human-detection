"""
Базовый интерфейс детектора.
"""

from typing import List, Tuple
import numpy as np


class BaseDetector:
    """Базовый класс, чтобы все детекторы имели одинаковый интерфейс"""

    class_names: List[str] = ["person"]  # один класс: человек

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Должен вернуть кортеж numpy-массивов: (boxes, scores, class_ids)
        - boxes: (N, 4) [x1, y1, x2, y2]
        - scores: (N,)
        - class_ids: (N,) — у нас всегда 0 (person), так как детектируем только людей
        """
        raise NotImplementedError
