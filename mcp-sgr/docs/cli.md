# 🧰 CLI команды

Запуск: `python -m src.cli --help`

- server — старт MCP‑SGR сервера
  - --stdio — MCP stdio транспорт
  - --http — HTTP фасад (FastAPI)
  - --port 8080 — порт HTTP

- analyze <task> — анализ задачи по SGR
  - --schema [auto|analysis|planning|decision|code_generation|summarization]
  - --budget [none|lite|full]
  - --json — вывести сырой JSON
  - --output <path> — сохранить вывод в файл

- enhance <prompt> — улучшение промпта
  - --level [minimal|standard|comprehensive]
  - --target <model>

- cache-stats — статистика кэша
- traces — последние трейсы (арг.: --limit, --tool)
- cleanup — очистка кэша/трейсов
- export-openapi — экспорт схемы OpenAPI (--format json|yaml, --output <path>)

Переменные окружения: см. [configuration.md](./configuration.md)