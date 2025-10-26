"""
Точка входа проекта.
Запускает детекцию людей на видео и сохраняет результаты.

1) Парсим аргументы через библиотеку human_detection.config
2) Передаём их в функцию human_detection.pipeline.run_pipeline
"""

from __future__ import annotations
from human_detection.config import parse_args, make_configs
from human_detection.pipeline import run_pipeline


args = parse_args()
cfgs = make_configs(args)

# Запускаем pipeline (чтение видео => инференс => отрисовка => запись => отчёты)
run_pipeline(cfgs)
