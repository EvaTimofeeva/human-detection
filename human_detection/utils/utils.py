"""
Вспомогательные функции
"""

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def create_dir(p: Path) -> None:
    """создаем папку, если ее нет"""
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class TimeMeter:
    """
    Класс для измерения времени обработки кадра.
    Считаем экспоненциальное среднее времени кадра (EMA) (для получения стабильного значения FPS)
    """

    alpha: float = 0.9  # Коэффициент сглаживания
    t_ema: Optional[float] = None  # Экспоненциальное скользящее среднее
    t_start: Optional[float] = None  # Метка времени

    def start(self) -> None:
        """Запоминаем старт времени"""
        self.t_start = time.perf_counter()

    def stop(self) -> float:
        """
        Останавливает отсчет, рассчитывает время, обновляет EMA и возвращает время.
        """
        if self.t_start is None:
            return 0.0

        dt = time.perf_counter() - self.t_start
        self.t_start = None

        # Обновление EMA
        if self.t_ema is None:
            self.t_ema = dt
        else:
            self.t_ema = self.alpha * self.t_ema + (1 - self.alpha) * dt
        return dt

    @property
    def fps(self) -> float:
        """
        Расчет сглаженного FPS:
        FPS = 1 / t
        Используем сглаженное для получения сглаженного FPS
        """
        if not self.t_ema or self.t_ema <= 0:
            return 0.0
        return 1.0 / self.t_ema
