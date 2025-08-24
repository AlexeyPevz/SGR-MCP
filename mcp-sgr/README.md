# MCP‑SGR Server

Универсальный middleware для прозрачного и управляемого мышления LLM‑агентов на базе Model Context Protocol (MCP) и Schema‑Guided Reasoning (SGR). Подключается к любым агентам как прослойка: обогащает запросы/ответы структурированным reasoning, даёт трассировки, метрики качества, кэш и умный роутинг на дешёвые/локальные модели.

- Работает с любым агентом без изменения его кода (Plug & Play)
- Прозрачные reasoning‑трассы для дебага и аудита
- Улучшает качество слабых и дешёвых моделей за счёт структурирования
- MCP‑интерфейс для IDE/агентов + HTTP‑фасад для no‑code/ботов
- Готовая нода для n8n (обёртка любого агентского шага по SGR‑протоколу)

## Почему MCP‑SGR

- MCP стандартизирует инструменты и среду взаимодействия
- SGR вводит строгие схемы мышления: анализ → план → решение → валидация
- В связке это даёт объяснимость, снижение стоимости и ускорение промпт‑инжиниринга

## Возможности

- `apply_sgr`: применить выбранную схему к задаче (анализ, планирование, решение)
- `wrap_agent_call`: обернуть вызов любого агента (pre/post reasoning, метрики качества)
- `enhance_prompt_with_sgr`: превратить простой промпт в структурированный
- `learn_schema_from_examples`: выучить новую схему из примеров (roadmap)
- Библиотека схем: analysis, planning, decision, search, code_generation, summarization
- SGR‑budget: глубина reasoning (none/lite/full), семплирование, адаптивный режим
- Кэширование reasoning‑паттернов и результатов
- Умный роутинг между моделями (локальные/облачные) по типу задачи/бюджету
- OpenTelemetry‑трассировка, структурные логи, метрики качества/уверенности
- PII‑редакция, маскирование, хранение приватных reasoning‑трасс

## Архитектура (высокоуровнево)

- **MCP Server**: инструменты (apply_sgr, wrap_agent_call, enhance_prompt, learn_schema), ресурсы (schema_library, policy), нотификации (trace events)
- **SGR Engine**: генерация reasoning через LLM‑клиент, валидация по JSON Schema/Pydantic, расчёт confidence, извлечение действий
- **Agent Connector**: HTTP/gRPC/локальный callable вызов исходного агента
- **Storage**: SQLite/DuckDB для кэша и трасс (старт), совместимо с внешними БД
- **Router**: политика выбора модели (локальная через Ollama/vLLM, внешняя через API)
- **HTTP‑фасад**: REST‑доступ к тем же функциям MCP для n8n/ботов/скриптов

## Структура репозитория

```
mcp-sgr/
├── src/
│   ├── server.py           # MCP сервер
│   ├── schemas/
│   │   ├── base.py         # базовые поля/валидаторы
│   │   ├── analysis.py     # анализ задач
│   │   ├── planning.py     # планирование
│   │   ├── decision.py     # принятие решений/валидация
│   │   ├── code.py         # кодогенерация
│   │   ├── summary.py      # суммаризация
│   │   └── custom.py       # пользовательские схемы
│   ├── tools/
│   │   ├── apply_sgr.py
│   │   ├── wrap_agent.py
│   │   ├── enhance_prompt.py
│   │   └── learn_schema.py
│   └── utils/
│       ├── validator.py
│       ├── llm_client.py
│       ├── router.py
│       ├── cache.py
│       ├── telemetry.py
│       └── redact.py
├── examples/
│   ├── basic_usage.py
│   ├── agent_wrapper.py
│   └── custom_schema.py
├── integrations/
│   ├── n8n-node/           # кастомная нода
│   ├── langchain/
│   ├── autogen/
│   └── crewai/
├── tests/
├── mcp.json                # MCP‑манифест
├── .env.example
├── README.md
└── Dockerfile
```

## Быстрый старт

### 1) Требования

- Python 3.11+
- Опционально: Docker
- Опционально: локальные модели (Ollama) или внешний провайдер API

### 2) Установка (локально)

```bash
# Клонировать репозиторий
git clone https://github.com/your-org/mcp-sgr
cd mcp-sgr

# Установить зависимости
pip install -e .

# Скопировать .env.example в .env и настроить ключи/эндпоинты
cp .env.example .env
```

### 3) Запуск MCP‑сервера

```bash
# Запуск MCP-сервера
python -m src.server

# По умолчанию доступны:
# - stdio‑транспорт (для IDE)
# - HTTP‑фасад на порту из .env
```

### 4) Проверка

```bash
# Выполнить примеры
python examples/basic_usage.py
python examples/agent_wrapper.py
```

## Конфигурация (ENV)

```env
# LLM провайдеры
LLM_BACKENDS=ollama,openrouter,vllm,custom
OLLAMA_HOST=http://localhost:11434
OPENROUTER_API_KEY=your-key
CUSTOM_LLM_URL=https://your-llm-endpoint

# SGR настройки
SGR_BUDGET_DEPTH=lite              # none|lite|full
SGR_PRE_ANALYSIS=auto              # auto|always|never
SGR_POST_ANALYSIS=lite             # lite|full
SGR_SAMPLE_RATE=0.5                # 0.0–1.0 (доля запросов с full‑режимом)

# Кэширование
CACHE_ENABLED=true
CACHE_STORE=sqlite:///./data/cache.db

# Трассировка
TRACE_ENABLED=true
PII_REDACT=true

# HTTP фасад
HTTP_PORT=8080
HTTP_AUTH_TOKEN=optional-token
```

## Политика роутинга (пример router_policy.yaml)

```yaml
router:
  rules:
    - when: task_type == "code_generation"
      use: qwen2.5-coder:7b@ollama
    - when: task_type in ["analysis","summarization"] and tokens < 4_000
      use: llama3.1:8b@ollama
    - when: risk == "high" or tokens >= 16_000
      use: cloud-default
  retry:
    max_attempts: 2
    backoff: 0.8
```

## MCP манифест (mcp.json)

```json
{
  "name": "sgr-reasoning",
  "description": "Structured reasoning middleware for any agent via MCP",
  "version": "1.0.0",
  "tools": [
    {
      "name": "apply_sgr",
      "description": "Apply SGR schema to analyze and structure a task",
      "input_schema": {
        "type": "object",
        "properties": {
          "task": { "type": "string", "description": "Task description" },
          "context": { "type": "object", "description": "Additional context" },
          "schema_type": { "type": "string", "enum": ["auto", "analysis", "planning", "decision", "search", "code_generation", "summarization"], "default": "auto" },
          "custom_schema": { "type": "object", "description": "Custom schema if schema_type is 'custom'" }
        },
        "required": ["task"]
      }
    },
    {
      "name": "wrap_agent_call",
      "description": "Wrap any agent call with pre/post SGR analysis",
      "input_schema": {
        "type": "object",
        "properties": {
          "agent_endpoint": { "type": "string" },
          "agent_request": { "type": "object" },
          "sgr_config": { "type": "object" }
        },
        "required": ["agent_endpoint", "agent_request"]
      }
    },
    {
      "name": "enhance_prompt_with_sgr",
      "description": "Enhance a simple prompt with SGR structure",
      "input_schema": {
        "type": "object",
        "properties": {
          "original_prompt": { "type": "string" },
          "target_model": { "type": "string" }
        },
        "required": ["original_prompt"]
      }
    },
    {
      "name": "learn_schema_from_examples",
      "description": "Learn new SGR schema from examples (roadmap feature)",
      "input_schema": {
        "type": "object",
        "properties": {
          "examples": { "type": "array", "items": { "type": "object" } },
          "task_type": { "type": "string" }
        },
        "required": ["examples", "task_type"]
      }
    }
  ],
  "resources": [
    {
      "name": "schema_library",
      "description": "Available SGR schemas",
      "uri": "sgr://schemas"
    },
    {
      "name": "policy",
      "description": "Current routing and budget policy",
      "uri": "sgr://policy"
    }
  ]
}
```

## Инструменты MCP (интерфейсы)

### apply_sgr

```python
result = await sgr.apply_sgr(
    task="Analyze user authentication flow for security issues",
    context={"codebase": "auth_module.py", "framework": "FastAPI"},
    schema_type="analysis"
)
# Возвращает:
# {
#   "reasoning": {...},      # JSON по схеме
#   "confidence": 0.85,      # 0..1
#   "suggested_actions": [...],
#   "metadata": {...}
# }
```

### wrap_agent_call

```python
result = await sgr.wrap_agent_call(
    agent_endpoint="coding_agent.generate_code",
    agent_request={"prompt": "создай REST API для блога"},
    sgr_config={
        "schema_type": "code_generation",
        "budget": "lite",
        "include_alternatives": True
    }
)
# Возвращает:
# {
#   "original_response": {...},
#   "reasoning_chain": {"pre": {...}, "post": {...}},
#   "quality_metrics": {...},
#   "suggestions": [...]
# }
```

### enhance_prompt_with_sgr

```python
enhanced = await sgr.enhance_prompt_with_sgr(
    original_prompt="Сделай скрипт бэкапа БД",
    target_model="llama3.1:8b@ollama"
)
# Возвращает улучшенный промпт со структурой
```

## Библиотека SGR‑схем

### Analysis Schema

```json
{
  "$id": "schema://analysis",
  "type": "object",
  "required": ["understanding", "goals", "constraints", "risks"],
  "properties": {
    "understanding": {
      "type": "object",
      "required": ["task_summary", "key_aspects"],
      "properties": {
        "task_summary": { "type": "string", "minLength": 10 },
        "key_aspects": { "type": "array", "items": { "type": "string" } },
        "ambiguities": { "type": "array", "items": { "type": "string" } }
      }
    },
    "goals": {
      "type": "object",
      "required": ["primary", "success_criteria"],
      "properties": {
        "primary": { "type": "string" },
        "secondary": { "type": "array", "items": { "type": "string" } },
        "success_criteria": { "type": "array", "items": { "type": "string" } }
      }
    },
    "constraints": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": { "enum": ["technical", "business", "resource", "time"] },
          "description": { "type": "string" }
        }
      }
    },
    "risks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "risk": { "type": "string" },
          "likelihood": { "enum": ["low", "medium", "high"] },
          "impact": { "enum": ["low", "medium", "high"] },
          "mitigation": { "type": "string" }
        }
      }
    }
  }
}
```

### Code Generation Schema

```json
{
  "$id": "schema://code_generation",
  "type": "object",
  "required": ["understanding", "design", "implementation", "validation"],
  "properties": {
    "understanding": {
      "type": "object",
      "required": ["goal", "constraints"],
      "properties": {
        "goal": { "type": "string", "minLength": 3 },
        "constraints": { "type": "array", "items": { "type": "string" } },
        "assumptions": { "type": "array", "items": { "type": "string" } }
      }
    },
    "design": {
      "type": "object",
      "properties": {
        "approach": { "type": "string" },
        "steps": { "type": "array", "items": { "type": "string" } },
        "alternatives": { "type": "array", "items": { "type": "string" } }
      }
    },
    "implementation": {
      "type": "object",
      "properties": {
        "language": { "type": "string", "enum": ["python", "js", "ts", "go", "java", "other"] },
        "framework": { "type": "string" },
        "code": { "type": "string" }
      }
    },
    "validation": {
      "type": "object",
      "properties": {
        "checks": { "type": "array", "items": { "type": "string" } },
        "test_plan": { "type": "array", "items": { "type": "string" } },
        "gaps": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

## Алгоритм confidence

Расчёт уверенности основан на:

1. **Полнота** (40%): доля заполненных обязательных полей схемы
2. **Согласованность** (30%): соответствие pre/post анализа
3. **Самопроверка** (20%): прохождение критериев успеха
4. **Валидность** (10%): успешная валидация по схеме

## Примеры использования

### 1) Базовое использование

```python
import asyncio
from src.client import SGRClient

async def main():
    sgr = SGRClient()
    
    # Применить схему анализа
    result = await sgr.apply_sgr(
        task="Оптимизировать производительность API",
        schema_type="analysis"
    )
    
    print(f"Confidence: {result['confidence']}")
    print(f"Risks: {result['reasoning']['risks']}")
```

### 2) Обёртка агента

```python
# Обернуть вызов существующего агента
result = await sgr.wrap_agent_call(
    agent_endpoint="http://localhost:8000/generate",
    agent_request={
        "prompt": "Создай микросервис для обработки платежей",
        "max_tokens": 2000
    },
    sgr_config={
        "schema_type": "code_generation",
        "budget": "full",
        "pre_analysis": True,
        "post_analysis": True
    }
)

# Анализ качества ответа
if result["quality_metrics"]["confidence"] < 0.7:
    print("Low confidence! Suggestions:")
    for suggestion in result["suggestions"]:
        print(f"- {suggestion}")
```

### 3) Улучшение промпта

```python
# Превратить простой промпт в структурированный
enhanced = await sgr.enhance_prompt_with_sgr(
    original_prompt="Напиши парсер логов",
    target_model="llama3.1:8b"
)

print(enhanced)
# Выведет структурированный промпт с четкими инструкциями
```

## Интеграция с n8n

### Установка ноды

1. Скопировать `integrations/n8n-node/` в папку кастомных нод n8n
2. Перезапустить n8n

### Использование в workflow

1. Добавить ноду "SGR Wrapper" перед/после AI нод
2. Настроить:
   - Server URL: `http://localhost:8080`
   - Mode: `Wrapper` / `Apply` / `Enhance`
   - Schema Type: `auto` / `analysis` / etc.
   - Budget: `lite` / `full`

### Пример workflow

```
[Trigger] → [Set Data] → [SGR Wrapper (pre)] → [OpenAI] → [SGR Wrapper (post)] → [IF confidence < 0.6] → [Notify]
```

## Телеметрия и мониторинг

### OpenTelemetry spans

- `sgr.pre_analysis`: предварительный анализ
- `sgr.agent_call`: вызов агента
- `sgr.post_analysis`: постанализ
- `sgr.validation`: валидация результата

### Метрики

- `sgr_confidence_distribution`: распределение confidence
- `sgr_cache_hit_rate`: процент попаданий в кэш
- `sgr_reasoning_duration`: время выполнения reasoning
- `sgr_model_usage`: использование моделей

## Безопасность

- PII редакция включается через `PII_REDACT=true`
- Приватные reasoning трассы хранятся отдельно
- Поддержка внешних secret managers для ключей
- Аудит лог всех операций

## Тестирование

```bash
# Запуск всех тестов
pytest

# Только юнит-тесты
pytest tests/unit/

# Интеграционные тесты
pytest tests/integration/

# С покрытием
pytest --cov=src --cov-report=html
```

## Дорожная карта

- [ ] LangChain/LangGraph middleware
- [ ] AutoGen GroupChat wrapper
- [ ] CrewAI per-agent SGR
- [ ] Schema learning from examples
- [ ] Private/public schema catalog
- [ ] Differential privacy for schemas
- [ ] Advanced observability dashboard
- [ ] A/B testing framework

## Вклад в проект

1. Fork репозитория
2. Создать feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Открыть Pull Request

### Требования к PR

- Проходят все тесты
- Добавлены новые тесты для новой функциональности
- Код отформатирован (black, isort)
- Обновлена документация

## Поддержка

- Issues: [GitHub Issues](https://github.com/your-org/mcp-sgr/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/mcp-sgr/discussions)
- Email: support@mcp-sgr.dev

## Лицензия

- Ядро: MIT License
- Enterprise features: отдельная лицензия

---

Сделано с ❤️ для сообщества AI-разработчиков