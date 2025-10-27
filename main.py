"""
Точка входа проекта.
Запускает детекцию людей на видео и сохраняет результаты.

1) Парсим аргументы через human_detection.config
2) Формируем конфигурации
3) Передаём их в human_detection.pipeline.run_pipeline
"""

from human_detection.config import parse_args, make_configs
from human_detection.pipeline import run_pipeline

# 1) Разбираем аргументы CLI
args = parse_args()
# 2) Собираем конфигурацию приложения
cfgs = make_configs(args)
# 3) Запускаем пайплайн:
#    - если выбран режим сравнения моделей, сначала сравним,
#      затем выполним финальный прогон лучшей моделью
run_pipeline(cfgs)
