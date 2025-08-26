# 📈 Наблюдаемость

## Телеметрия
- Опциональный OpenTelemetry экспорт (OTLP gRPC)
- Включение: `OTEL_ENABLED=true`
- Эндпоинт: `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`

## Трейсы reasoning
- Хранятся в SQLite (по умолчанию)
- Просмотр: `python -m src.cli traces --limit 10`

## Дашборды
- Рекомендуется Grafana + Tempo/Jaeger (см. docker-compose профили)