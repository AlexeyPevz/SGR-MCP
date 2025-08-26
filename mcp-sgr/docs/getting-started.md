# 🚀 Быстрый старт

## Требования
- Python 3.11+
- (Опционально) Docker и Docker Compose

## Установка (локально)
```bash
# Установка зависимостей (рекомендуется venv)
pip install -U pip
pip install -e .[dev]

# Проверка
pytest -q
```

## Запуск HTTP API
```bash
export HTTP_REQUIRE_AUTH=true
export HTTP_AUTH_TOKEN=change-me
python -m src.cli server --http --port 8080
```
- OpenAPI: см. `python -m src.cli export-openapi --format json` или [HTTP API](./http-api.md)

## CLI примеры
```bash
# Анализ задачи
python -m src.cli analyze "Design a todo REST API" --schema planning --json

# Улучшение промпта
python -m src.cli enhance "Write a Python function" --level standard

# Экспорт OpenAPI
python -m src.cli export-openapi --format yaml --output openapi.yaml
```

## Docker / Compose
```bash
# Сборка образа
docker build -t mcp-sgr:latest .

# Быстрый запуск инфраструктуры
OPENROUTER_API_KEY=... docker compose up -d
```
Подробнее: [deployment.md](./deployment.md)