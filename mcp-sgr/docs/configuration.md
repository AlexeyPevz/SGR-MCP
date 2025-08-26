# ⚙️ Конфигурация

## Общие
- LLM_BACKENDS=ollama,openrouter
- OLLAMA_HOST=http://localhost:11434
- OPENROUTER_API_KEY=...

## HTTP
- HTTP_ENABLED=true
- HTTP_PORT=8080
- HTTP_HOST=0.0.0.0
- HTTP_REQUIRE_AUTH=true
- HTTP_AUTH_TOKEN=change-me
- HTTP_CORS_ORIGINS=*

## Кэш и трейсы
- CACHE_ENABLED=true
- CACHE_STORE=sqlite:///./data/cache.db
- CACHE_TTL_SECONDS=3600
- TRACE_ENABLED=true
- TRACE_STORE=sqlite:///./data/traces.db
- TRACE_RETENTION_DAYS=7

## Телеметрия
- OTEL_ENABLED=false
- OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

## Маршрутизация
- ROUTER_POLICY_FILE=./config/router_policy.yaml
- ROUTER_DEFAULT_BACKEND=ollama