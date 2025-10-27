"""
1) Читаем видео
1.1) Можем сравнивать несколько моделей и выбирать лучшую
2) Запускаем детектор
3) Рисуем результаты детектора
4) Пишем результаты детектора в видео
5) Пишем отчёты в CSV и JSON
"""

from pathlib import Path
from typing import Dict, Tuple

import csv
import cv2
from tqdm import tqdm

from .config import AppConfig
from .models.detector import PeopleDetector, DetectorConfig
from .utils.utils import TimeMeter, create_dir
from .metrics.report import Report, FrameStats
from .video import open_reader, open_writer
from .visualize import draw_detections


def create_dirs(cfg: AppConfig) -> None:
    """создаем папку, если ее нет"""
    create_dir(cfg.output_path.parent)
    create_dir(cfg.metrics_dir)


def run_model(
    cfg: AppConfig, model_name: str, suffix_to_output: bool = True
) -> Tuple[Report, Path]:
    """Запускаем обработку видео одной моделью, возвращаем Report и путь к выводу видео."""

    # Подготовка конфигов и путей
    det_cfg = DetectorConfig(
        model=model_name,
        model_path=cfg.detector.model_path,
        conf=cfg.detector.conf,
        iou=cfg.detector.iou,
        imgsz=cfg.detector.imgsz,
        device=cfg.detector.device,
        half=cfg.detector.half,
        classes=cfg.detector.classes,
    )

    out_path = cfg.output_path
    if suffix_to_output:
        out_path = out_path.with_name(f"{out_path.stem}_{model_name}{out_path.suffix}")

    # Детектор и I/O
    detector = PeopleDetector(det_cfg)
    class_names = detector.class_names

    cap, spec = open_reader(cfg.input_path)
    writer = open_writer(
        out_path, spec, fourcc_fallback="mp4v", fps_override=cfg.fps_override
    )

    report = Report()
    timer = TimeMeter()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = cfg.max_frames if cfg.max_frames is not None else (total or None)
    pbar_total = limit if limit is not None else (total if total > 0 else None)

    print(f"{model_name}: Conf: {cfg.detector.conf}  IoU: {cfg.detector.iou}")
    pbar = tqdm(total=pbar_total, desc=f"{model_name}: обработка", unit="кадр")
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if limit is not None and frame_idx >= limit:
                break

            # Инференс
            timer.start()
            boxes, scores, cls_ids = detector.infer(frame)
            dt = timer.stop()

            # Отрисовка и запись
            if boxes is not None and len(boxes) > 0:
                frame = draw_detections(
                    frame, boxes, scores, cls_ids, class_names, cfg.viz
                )
            writer.write(frame)

            # Статистика
            persons = int((cls_ids == 0).sum()) if cls_ids is not None else 0
            report.add(
                FrameStats(
                    frame_idx=frame_idx,
                    detections=int(len(cls_ids)) if cls_ids is not None else 0,
                    persons=persons,
                    fps_inst=(1.0 / dt) if dt > 0 else 0.0,
                )
            )

            # Предпросмотр
            if cfg.show:
                cv2.imshow(f"human_detection[{model_name}] (q = выход)", frame)
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

    # Сохранение отчётов именно для этой модели
    stem = out_path.stem
    report.flush(out_dir=cfg.metrics_dir, stem=stem)

    return report, out_path


def score_model(summary: Dict[str, float]) -> float:
    """
    Эвристический скор модели для выбора лучшей. Сравниваем по:
      - количество детектируемых людей в среднем
      - FPS
    """
    persons_per_frame = summary["persons"] / max(1.0, summary["frames"])
    fps = summary["fps_mean"]
    # веса: качество важнее скорости
    return persons_per_frame * 1.0 + fps * 0.2


def compare_models_and_pick_best(cfg: AppConfig) -> str:
    """
    - Запускаем сравнение моделей
    - Сохраняет таблицу метрик в metrics,
     - Возвращаем (лучшую модель, список (model, summary_dict))
    """
    rows = []
    for model in cfg.compare_backends:
        rep, _ = run_model(cfg, model_name=model, suffix_to_output=True)
        s = rep.summarize()
        score = score_model(s)
        rows.append((model, s, score))

    # сохраняем таблицу
    table_path = cfg.metrics_dir / "models_comparison.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "frames",
                "detections",
                "persons",
                "fps_mean",
                "runtime_sec",
                "score",
            ]
        )
        for model, s, score in rows:
            w.writerow(
                [
                    model,
                    int(s["frames"]),
                    int(s["detections"]),
                    int(s["persons"]),
                    f"{s['fps_mean']:.3f}",
                    f"{s['runtime_sec']:.2f}",
                    f"{score:.3f}",
                ]
            )
    print(f"Таблица сохранена: {table_path}")

    # выбираем лучшую модель
    best_model = max(rows, key=lambda x: x[2])[0]
    print(f"Лучшая модель: {best_model}")

    return best_model


def run_pipeline(cfg: AppConfig) -> None:
    """
    Главный пайплайн.
    Если compare_models=True или model='auto', сначала сравниваем модели,
    затем запускаем финальный прогон с лучшей моделью с именем выхода, заданным пользователем.
    """
    create_dirs(cfg)

    if cfg.compare_models:
        print("Процесс сравнения моделей:")
        compare_models_and_pick_best(cfg)
        # финальный прогон без суффикса, чтобы результат назывался как пользователь указал
    else:
        run_model(cfg, model_name=cfg.detector.model, suffix_to_output=False)
