# Интеграция MCP-SGR с n8n (без custom node)

## Быстрый старт

MCP-SGR предоставляет HTTP API, который можно использовать в n8n через стандартную HTTP Request ноду.

## 1. Запуск MCP-SGR сервера

На вашем VPS или локально:

```bash
cd mcp-sgr

# С Gemini (если используете Gemini CLI локально)
export GEMINI_API_KEY="your-key"
python examples/gemini_proxy.py &
python -m src.http_server

# Или с Ollama
python -m src.http_server
```

## 2. Доступные эндпоинты для n8n

### POST /v1/apply-sgr
Применить схему анализа к задаче.

**Request:**
```json
{
  "task": "Analyze database performance issues",
  "schema_type": "analysis",
  "budget": "full",
  "context": {
    "database": "PostgreSQL",
    "size": "100GB"
  }
}
```

**Response:**
```json
{
  "reasoning": { ... },
  "confidence": 0.85,
  "suggested_actions": ["Add indexes", "Optimize queries"],
  "metadata": { ... }
}
```

### POST /v1/enhance-prompt
Улучшить промпт для LLM.

**Request:**
```json
{
  "original_prompt": "Write API documentation",
  "target_model": "gemini-pro"
}
```

### POST /v1/wrap-agent
Обернуть вызов другого агента/API.

**Request:**
```json
{
  "agent_endpoint": "http://your-agent/api",
  "agent_request": { ... },
  "sgr_config": {
    "schema_type": "code_generation",
    "pre_analysis": true
  }
}
```

## 3. Настройка в n8n

### Вариант A: HTTP Request нода

1. Добавьте **HTTP Request** ноду
2. Настройте:
   - **Method**: POST
   - **URL**: `http://your-server:8080/v1/apply-sgr`
   - **Authentication**: None (или Header Auth если включена)
   - **Send Headers**: 
     - Content-Type: application/json
     - X-API-Key: your-key (если нужно)
   - **Send Body**: JSON с параметрами

### Вариант B: Code нода (JavaScript)

```javascript
const response = await $http.request({
  method: 'POST',
  url: 'http://your-server:8080/v1/apply-sgr',
  body: {
    task: items[0].json.task,
    schema_type: 'analysis',
    budget: 'full',
    context: items[0].json.context
  },
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': $credentials.sgrApiKey // если используете credentials
  }
});

return [{
  json: {
    confidence: response.data.confidence,
    actions: response.data.suggested_actions,
    reasoning: response.data.reasoning
  }
}];
```

## 4. Примеры workflow

### Анализ и принятие решений

1. **Trigger** → получаем задачу
2. **HTTP Request** → POST /v1/apply-sgr (schema_type: "analysis")
3. **IF** → проверяем confidence > 0.7
4. **HTTP Request** → POST /v1/apply-sgr (schema_type: "decision")
5. **Set** → формируем результат

### Улучшение промптов для ChatGPT

1. **Webhook** → получаем простой промпт
2. **HTTP Request** → POST /v1/enhance-prompt
3. **OpenAI** → используем улучшенный промпт
4. **HTTP Request** → POST /v1/apply-sgr для анализа ответа

### Мониторинг качества AI ответов

1. **Schedule Trigger** → каждый час
2. **HTTP Request** → GET /v1/traces?limit=100
3. **Code** → анализируем confidence метрики
4. **IF** → если среднее < 0.6
5. **Email** → отправляем алерт

## 5. Credentials в n8n

Создайте Generic Credential:
- **Name**: MCP-SGR API
- **Generic Auth Type**: Header Auth
- **Header Name**: X-API-Key
- **Header Value**: your-api-key

## 6. Обработка ошибок

В HTTP Request ноде включите:
- **Continue On Fail**: true
- **Include Response Headers and Status**: true

Затем добавьте IF ноду для проверки:
```javascript
{{ $json.responseCode === 200 }}
```

## 7. Полезные expressions

```javascript
// Извлечь топ-3 действия
{{ $json.suggested_actions.slice(0, 3).join('\n') }}

// Проверить высокую уверенность
{{ $json.confidence >= 0.8 ? 'High' : 'Low' }}

// Форматировать reasoning
{{ JSON.stringify($json.reasoning, null, 2) }}
```

## 8. Мониторинг

Добавьте в workflow:
1. **HTTP Request** → GET /v1/cache-stats
2. **Postgres** → сохранить метрики
3. **Grafana** → визуализация

## Преимущества HTTP интеграции

✅ Не нужно писать custom node
✅ Работает из коробки
✅ Легко отлаживать
✅ Можно использовать в cloud n8n
✅ Поддержка всех функций MCP-SGR

## Troubleshooting

1. **Connection refused**
   - Проверьте что сервер запущен
   - Проверьте firewall правила
   - Используйте правильный IP/порт

2. **401 Unauthorized**
   - Добавьте X-API-Key header
   - Проверьте значение HTTP_AUTH_TOKEN

3. **Timeout**
   - Увеличьте timeout в HTTP Request ноде
   - Используйте budget: "lite" для быстрых ответов