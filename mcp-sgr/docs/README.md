# 📚 Документация MCP‑SGR

Добро пожаловать в централизованную документацию проекта MCP‑SGR (Schema‑Guided Reasoning). Здесь собраны основные материалы, объединённые по темам.

## Быстрый старт
- [Установка и запуск](./getting-started.md)
- [CLI команды](./cli.md)
- [HTTP API (OpenAPI)](./http-api.md)

## Основы и архитектура
- [Концепции SGR и схемы](./sgr-concepts.md)
- [Архитектура и структура проекта](./architecture.md)

## Эксплуатация
- [Конфигурация (переменные окружения)](./configuration.md)
- [Деплой (Docker/Docker Compose)](./deployment.md)
- [Наблюдаемость и телеметрия](./observability.md)

## Качество и развитие
- [Бенчмарки и результаты](./benchmarks.md)
- [Вклад в проект (Contributing)](./contributing.md)

## Полезные ссылки
- Экспорт OpenAPI из CLI: `python -m src.cli export-openapi --format json --output openapi.json`
- Файл политики роутинга моделей: `mcp-sgr/router_policy.yaml`

---

Примечание: исторические и обзорные документы из корня (`*.md`) сохранены для справки. Там, где есть пересечение, они помечены ссылкой на актуальные страницы здесь.

