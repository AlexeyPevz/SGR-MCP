from src.utils.router import ModelRouter, TaskType


def test_detect_task_type_russian_keywords():
    r = ModelRouter()
    assert r.detect_task_type("Сделай анализ требований") == TaskType.ANALYSIS
    assert r.detect_task_type("Подготовь план развертывания") == TaskType.PLANNING
    assert r.detect_task_type("Реши, что выбрать: Postgres или Mongo") == TaskType.DECISION
    assert r.detect_task_type("Напиши код для парсера") == TaskType.CODE_GENERATION
    assert r.detect_task_type("Сделай краткое резюме документа") == TaskType.SUMMARIZATION
    assert r.detect_task_type("Найди ошибку в логе") == TaskType.SEARCH

