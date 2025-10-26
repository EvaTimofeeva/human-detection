"""
Работа с видео через OpenCV: читаем и записываем видео, с аккуратным выбором кодека
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

import cv2


@dataclass
class VideoSpec:
    """Параметры видео"""

    width: int
    height: int
    fps: float
    fourcc: int  # исходный fourcc видео (может быть 0, если не определён)


def _safe_fps(fps: float) -> float:
    """Возвращает валидный FPS (по умолчанию 25.0, если пришёл ноль/NaN)."""
    try:
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    return 25.0


def open_reader(path: Path) -> Tuple[cv2.VideoCapture, VideoSpec]:
    """
    Открываем видео на чтение и возвращаем (cap, VideoSpec).

    :param path: путь к входному видеофайлу
    """
    if not path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    # извлекаем метаданные
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = _safe_fps(float(cap.get(cv2.CAP_PROP_FPS)))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # может быть 0

    # если размер не прочитался — считываем один кадр для инициализации
    if width == 0 or height == 0:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Не удалось прочитать первый кадр из: {path}")
        height, width = frame.shape[:2]
        # возвращаемся на начало, если нужно дальше читать с нуля
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    spec = VideoSpec(width=width, height=height, fps=fps, fourcc=fourcc)
    return cap, spec


def open_writer(
    path: Path,
    video_spec: VideoSpec,
    fourcc_fallback: str = "mp4v",
    fps_override: Optional[float] = None,
) -> cv2.VideoWriter:
    """
    Создаём видеописатель с максимально совместимыми настройками.
    На Windows для .avi используем MJPG; для .mp4 — mp4v.
    При неудаче пробуем резервный вариант AVI+MJPG.

    :param path: путь для сохранения выходного видео
    :param video_spec: спецификация входного видео
    :param fourcc_fallback: запасной кодек (строка fourcc), по умолчанию 'mp4v'
    :param fps_override: принудительный FPS на выходе
    :return: открытый cv2.VideoWriter
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Выбираем "лучший шанс" кодека по расширению файла
    ext = path.suffix.lower()
    is_windows = os.name == "nt"

    if ext == ".avi":
        # На Windows AVI+MJPG почти всегда дружит с OpenCV
        primary_fourcc_str = "MJPG" if is_windows else "MJPG"
    elif ext == ".mp4":
        # Для MP4 не лезем в h264/openh264; берём более совместимый mp4v
        primary_fourcc_str = "mp4v"
    else:
        # Непривычное расширение — используем то, что попросили как fallback
        primary_fourcc_str = fourcc_fallback

    primary_fourcc = cv2.VideoWriter_fourcc(*primary_fourcc_str)
    fps = (
        fps_override
        if (fps_override and fps_override > 0)
        else _safe_fps(video_spec.fps)
    )

    # Первая попытка — выбранный кодек и исходное имя
    writer = cv2.VideoWriter(
        str(path), primary_fourcc, fps, (video_spec.width, video_spec.height)
    )

    if writer.isOpened():
        return writer

    # Вторая попытка — явный fallback-кодек с тем же путём
    fallback_fourcc = cv2.VideoWriter_fourcc(*fourcc_fallback)
    writer = cv2.VideoWriter(
        str(path), fallback_fourcc, fps, (video_spec.width, video_spec.height)
    )
    if writer.isOpened():
        print(f"[info] Переключился на fourcc={fourcc_fallback} для файла: {path}")
        return writer

    # Третья попытка: AVI + MJPG (меняем расширение)
    alt_path = path.with_suffix(".avi")
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(alt_path), mjpg, fps, (video_spec.width, video_spec.height)
    )
    if writer.isOpened():
        print(
            f"[info] MP4/выбранный кодек не завёлся. Записываю как AVI/MJPG: {alt_path}"
        )
        return writer
    raise RuntimeError(
        "Не удалось инициализировать VideoWriter. "
        f"Пробовал: {path} с fourcc={primary_fourcc_str}, затем fourcc={fourcc_fallback}, "
        f"затем {alt_path} с fourcc=MJPG. "
        "На Windows для MP4/H.264 требуется openh264 DLL; проще писать AVI/MJPG."
    )
