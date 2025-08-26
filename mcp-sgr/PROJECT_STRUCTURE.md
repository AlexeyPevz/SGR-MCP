# 📁 Структура проекта MCP-SGR

## 🏗️ Организация файлов

```
/workspace/mcp-sgr/
│
├── 📄 README.md                           # Главный README проекта
├── 📊 CONSOLIDATED_BENCHMARK_RESULTS.md   # Все результаты в одном месте
├── 🏆 FINAL_SGR_BENCHMARK_REPORT.md       # Финальный отчет с выводами
├── 📈 PROJECT_STATUS.md                   # Текущий статус проекта
├── 📁 PROJECT_STRUCTURE.md                # Этот файл
│
├── 📦 benchmark-pack/                     # Основной пакет бенчмаркинга
│   ├── 📄 README.md                      # Документация пакета
│   ├── 📄 BENCHMARK_PACK_OVERVIEW.md     # Обзор возможностей
│   │
│   ├── 📁 configs/                       # Конфигурации для тестов
│   │   ├── config.yaml                   # Базовая конфигурация
│   │   ├── config_extended.yaml          # Расширенная конфигурация
│   │   └── config_*.yaml                 # Другие конфигурации
│   │
│   ├── 📁 scripts/                       # Исполняемые скрипты
│   │   ├── benchmark_runner.py           # Основной скрипт запуска
│   │   ├── generate_comparison_tables.py # Генерация таблиц
│   │   ├── analyze_sgr_impact.py         # Анализ влияния SGR
│   │   ├── visualize_results.py          # Визуализация результатов
│   │   └── monitor_*.py/sh               # Скрипты мониторинга
│   │
│   ├── 📁 tasks/                         # Задачи для тестирования
│   │   ├── code_generation.yaml         # Задачи генерации кода
│   │   ├── rag_qa.yaml                  # Задачи RAG QA
│   │   ├── summarization.yaml           # Задачи суммаризации
│   │   ├── planning_decision.yaml       # Задачи планирования
│   │   ├── data_etl.yaml                # Задачи ETL
│   │   └── agent_workflow.yaml          # Задачи workflow
│   │
│   ├── 📁 eval/                          # Модули оценки
│   │   ├── __init__.py
│   │   └── metrics.py                    # Метрики оценки
│   │
│   ├── 📁 reports/                       # Результаты тестов
│   │   ├── benchmark_results_*.json      # JSON результаты
│   │   └── benchmark_report_*.md         # Markdown отчеты
│   │
│   ├── 📄 SGR_COMPARISON_TABLES_*.md     # Сравнительные таблицы
│   └── 📄 SGR_IMPACT_ANALYSIS_*.md       # Анализ влияния
│
├── 📦 sgr/                               # Основная библиотека SGR
│   ├── __init__.py                       # Инициализация модуля
│   ├── prompts.py                        # SGR промпты
│   ├── schemas.py                        # SGR схемы
│   ├── core.py                           # Основная логика
│   └── utils.py                          # Утилиты
│
├── 📦 benchmarks/                        # Дополнительные бенчмарки
│   └── README.md                         # Документация
│
├── 📦 archive/                           # Архив старых файлов
│   ├── old_experiments/                  # Старые эксперименты
│   └── old_reports/                      # Старые отчеты
│
└── 📁 reports/                           # Глобальные отчеты
    └── poc_*.md/json                     # POC отчеты
```

## 🔑 Ключевые файлы

### Документация
- `README.md` - Главная документация проекта
- `CONSOLIDATED_BENCHMARK_RESULTS.md` - Все результаты тестов
- `FINAL_SGR_BENCHMARK_REPORT.md` - Финальные выводы

### Основные скрипты
- `benchmark-pack/scripts/benchmark_runner.py` - Запуск тестов
- `benchmark-pack/scripts/generate_comparison_tables.py` - Генерация таблиц
- `benchmark-pack/scripts/analyze_sgr_impact.py` - Анализ результатов

### Конфигурации
- `benchmark-pack/configs/config.yaml` - Базовая конфигурация
- `benchmark-pack/configs/config_extended.yaml` - Полная конфигурация

### Задачи
- `benchmark-pack/tasks/*.yaml` - Все тестовые задачи по категориям

## 🚀 Как использовать

1. **Для запуска тестов**: Используйте скрипты из `benchmark-pack/scripts/`
2. **Для анализа результатов**: Смотрите отчеты в корне и `benchmark-pack/reports/`
3. **Для интеграции SGR**: Используйте модули из `sgr/`
4. **Для настройки тестов**: Редактируйте файлы в `benchmark-pack/configs/`

## 📊 Результаты находятся в:

- Сырые данные: `benchmark-pack/reports/benchmark_results_*.json`
- Отчеты: `benchmark-pack/reports/benchmark_report_*.md`
- Сводные таблицы: `benchmark-pack/SGR_COMPARISON_TABLES_*.md`
- Анализ: `benchmark-pack/SGR_IMPACT_ANALYSIS_*.md`
- Финальный отчет: `FINAL_SGR_BENCHMARK_REPORT.md`

## 🧹 Поддержание порядка

1. Новые эксперименты создавайте в `benchmark-pack/`
2. Старые файлы перемещайте в `archive/`
3. Результаты храните в `reports/`
4. Документацию обновляйте в корне проекта