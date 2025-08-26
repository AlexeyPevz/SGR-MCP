# 🌐 HTTP API

- Базовые переменные:
  - HTTP_REQUIRE_AUTH=true
  - HTTP_AUTH_TOKEN=... (используйте заголовок x-api-key)
- Базовый URL: http://localhost:8080

## Аутентификация
Передавайте заголовок: `x-api-key: <HTTP_AUTH_TOKEN>`

## Рейткеп
Простой in‑memory RPM ограничитель (можно отключить/настроить через RATE_LIMIT_*).

## Эндпоинты

- GET /health — состояние сервиса
- POST /v1/apply-sgr — применить SGR к задаче
- POST /v1/wrap-agent — обёртка вызова агента
- POST /v1/enhance-prompt — улучшение промпта
- POST /v1/learn-schema — обучение схемы (R&D)
- GET /v1/schemas — список доступных схем
- GET /v1/cache-stats — статистика кэша
- GET /v1/traces — последние трейсы (?limit=10&tool=...)

## OpenAPI
Экспорт схемы:
```bash
python -m src.cli export-openapi --format json --output openapi.json
```