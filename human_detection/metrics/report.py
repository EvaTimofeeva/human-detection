"""
Собираем статистику по кадрам и вносим в CSV/JSON.
Метрики: количество детекций/людей и средний FPS.
"""

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class FrameStats:
    """
    Класс-контейнер для хранения статистики по одному обработанному кадру
    """

    frame_idx: int
    detections: int
    persons: int
    fps_inst: float


@dataclass
class Report:
    """
    Основной класс для сбора и составления отчетов по всем кадрам
    """

    rows: List[FrameStats] = field(default_factory=list)
    started_at: float = field(
        default_factory=time.time
    )  # момент создания объекта Report, чтобы расчитать общее время работы

    def add(self, fs: FrameStats) -> None:
        """
        Добавляем статистику в список
        """
        self.rows.append(fs)

    def summarize(self) -> Dict[str, float]:
        """
        Агрегируем собранные данные и вычисляет сводную статистику.

        :return: Словарь с общими метриками.
        """

        total_frames = len(self.rows)
        total_det = sum(
            r.detections for r in self.rows
        )  # Сумма детекций по всем кадрам
        total_persons = sum(r.persons for r in self.rows)  # Сумма обнаруженных людей

        # Получаем список значений FPS, исключая нули (где время не удалось померить)
        fps_vals = [r.fps_inst for r in self.rows if r.fps_inst > 0]
        mean_fps = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0

        return {
            "frames": float(total_frames),
            "detections": float(total_det),
            "persons": float(total_persons),
            "fps_mean": float(mean_fps),
            "runtime_sec": float(time.time() - self.started_at),
        }

    def flush(self, out_dir: Path, stem: str):
        """
        Сохраняет данные в CSV и сводную статистику в JSON.

        :param out_dir: путь для сохранения отчетов.
        :param stem: имя файла
        :return: Кортеж (путь к CSV, путь к JSON).
        """
        # если нет пути
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{stem}.csv"
        json_path = out_dir / f"{stem}.json"

        # CSV
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "detections", "persons", "fps_inst"])
            for r in self.rows:
                w.writerow(
                    [r.frame_idx, r.detections, r.persons, f"{r.fps_inst:.3f}"]
                )  # записываем данных по каждому кадру

        # JSON
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self.summarize(), f, ensure_ascii=False, indent=2)

        return csv_path, json_path
