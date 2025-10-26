"""
1) Читаем видео
2) Запускаем детектор
3) Рисуем результаты детектора
4) Пишем результаты детектора в видео
5) Пишем отчёты в CSV и JSON
"""

import cv2
from tqdm import tqdm

from .config import AppConfig
from .detector import PeopleDetector
from .utils.utils import TimeMeter, ensure_dir
from .metrics.report import Report, FrameStats
from .video import open_reader, open_writer
from .visualize import draw_detections


def _prepare_dirs(cfg: AppConfig) -> None:
    """создаем папку, если ее нет"""
    ensure_dir(cfg.output_path.parent)
    ensure_dir(cfg.metrics_dir)


def run_pipeline(cfg: AppConfig) -> None:
    """Это главный пайплайн. Здесь все вычисления."""

    # инициализирую модель
    detector = PeopleDetector(cfg.detector)
    class_names = detector.class_names

    # открываю входное видео и подготавливаю писатель
    cap, spec = open_reader(cfg.input_path)
    writer = open_writer(
        cfg.output_path, spec, fourcc_fallback="mp4v", fps_override=cfg.fps_override
    )

    # отчёт
    report = Report()
    stem = cfg.output_path.stem

    # для удобства считаем FPS
    timer = TimeMeter()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = cfg.max_frames if cfg.max_frames is not None else total or None
    pbar_total = limit if limit is not None else total if total > 0 else None

    print(
        f"Модель: {cfg.detector.model_path}  Устройство: {cfg.detector.device}  "
        f"Conf: {cfg.detector.conf}  IoU: {cfg.detector.iou}  ImgSz: {cfg.detector.imgsz}"
    )

    pbar = tqdm(total=pbar_total, desc="Обработка", unit="кадр")
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit is not None and frame_idx >= limit:
                break

            # 4) инференс
            timer.start()
            boxes, scores, cls_ids = detector.infer(frame)
            dt = timer.stop()

            # 5) рисую и пишу в видео
            frame = draw_detections(frame, boxes, scores, cls_ids, class_names, cfg.viz)
            writer.write(frame)

            # 6) иногда сохраняю кадры
            if cfg.save_frames and (frame_idx % max(1, cfg.save_every) == 0):
                frames_dir = cfg.output_path.parent / "frames"
                ensure_dir(frames_dir)
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), frame)

            # 7) добавляю статистику
            persons = int((cls_ids == 0).sum())
            report.add(
                FrameStats(
                    frame_idx=frame_idx,
                    detections=len(cls_ids),
                    persons=persons,
                    fps_inst=(1.0 / dt) if dt > 0 else 0.0,
                )
            )

            # 8) предпросмотр
            if cfg.show:
                cv2.imshow("human_detection (q = выход)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
        if cfg.show:
            cv2.destroyAllWindows()

    # сохраняем отчёты
    csv_path, json_path = report.flush(out_dir=cfg.metrics_dir, stem=stem)
    summary = report.summarize()

    print("Готово.")
    print(f"Видео:   {cfg.output_path}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    print(
        f"Сводка:  frames={int(summary['frames'])}, "
        f"detections={int(summary['detections'])}, persons={int(summary['persons'])}, "
        f"fps_mean={summary['fps_mean']:.2f}, runtime={summary['runtime_sec']:.1f}s"
    )
