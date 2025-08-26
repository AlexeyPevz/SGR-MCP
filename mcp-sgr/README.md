# 🚀 MCP-SGR: Schema-Guided Reasoning для LLM

## 📋 О проекте

MCP-SGR — middleware для управляемого и объяснимого мышления LLM на основе Schema‑Guided Reasoning (SGR).

## 📚 Документация (актуальная)
Смотри централизованные материалы в `docs/`:
- [Быстрый старт](./docs/getting-started.md)
- [CLI команды](./docs/cli.md)
- [HTTP API и OpenAPI](./docs/http-api.md)
- [Архитектура](./docs/architecture.md)
- [Конфигурация](./docs/configuration.md)
- [Деплой](./docs/deployment.md)
- [Наблюдаемость](./docs/observability.md)
- [Бенчмарки](./docs/benchmarks.md)
- [Contributing](./docs/contributing.md)

Исторические отчёты и обзоры в корне сохранены для справки: они могут дублировать содержание и будут постепенно упраздняться.

## 🏁 Коротко о запуске
```bash
# Установка
pip install -U pip
pip install -e .[dev]

# Запуск HTTP API
export HTTP_REQUIRE_AUTH=true
export HTTP_AUTH_TOKEN=change-me
python -m src.cli server --http --port 8080

# Экспорт OpenAPI
python -m src.cli export-openapi --format json --output openapi.json
```

## 📄 Лицензия
MIT