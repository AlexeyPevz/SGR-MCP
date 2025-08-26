# 🚢 Деплой

## Docker
```bash
docker build -t mcp-sgr:latest .
```

## Docker Compose (локально)
```bash
export OPENROUTER_API_KEY=...
docker compose up -d
```
- Сервис `mcp-sgr` на порту 8080
- Сервис `ollama` подтягивает модели при старте
- Профили `observability`, `distributed` — для OTEL/Redis/Postgres

## Переменные окружения
См. [configuration.md](./configuration.md)