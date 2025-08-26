# 🏗️ Архитектура и структура проекта

## Компоненты
- MCP сервер (`src/server.py`) — инструменты и ресурсы MCP
- HTTP фасад (`src/http_server.py`) — REST API поверх SGR
- Схемы SGR (`src/schemas/*`) — доменные структуры и валидация
- Инструменты (`src/tools/*`) — apply_sgr, wrap_agent, enhance_prompt
- Утилиты (`src/utils/*`) — LLM клиент, кэш/трейсы, телеметрия, роутер

## Деплой
- Docker multi-stage образ (`Dockerfile`)
- docker-compose.yml — локальный стек (Ollama, опционально Redis/Postgres/OTEL)

## Маршрутизация моделей
- Политика: `router_policy.yaml`
- Детекция типа задачи (`utils/router.py`), выбор бэкенда и модели

## Наблюдаемость
- OpenTelemetry (опционально), кэш и трейсы в SQLite по умолчанию

Подробнее: [sgr-concepts.md](./sgr-concepts.md), [deployment.md](./deployment.md)