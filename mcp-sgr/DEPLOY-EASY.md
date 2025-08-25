# 🚀 САМОЕ ПРОСТОЕ РАЗВЕРТЫВАНИЕ MCP-SGR

## Вариант 1: СУПЕР-БЫСТРЫЙ (1 команда)

```bash
curl -sSL https://raw.githubusercontent.com/mcp-sgr/mcp-sgr/main/deploy-simple.sh | bash
```

Всё! Сервер запущен на порту 8080.

## Вариант 2: DOCKER (тоже 1 команда)

```bash
docker run -p 8080:8080 -e OPENAI_API_KEY=your-key mcp-sgr/mcp-sgr:latest
```

## Вариант 3: РУЧНОЙ (3 шага)

```bash
# 1. Клонировать
git clone https://github.com/mcp-sgr/mcp-sgr && cd mcp-sgr

# 2. Установить
pip install fastapi uvicorn httpx aiohttp pydantic

# 3. Запустить
python3 -m src.http_server
```

## 🎯 ЧТО ПОЛУЧАЕТЕ СРАЗУ:

### REST API на порту 8080:

```bash
# Тест что работает
curl http://localhost:8080/health

# Анализ задачи
curl -X POST http://localhost:8080/v1/apply-sgr \
  -H "Content-Type: application/json" \
  -d '{"task": "Оптимизировать запрос к БД", "schema_type": "analysis"}'
```

## 🔧 МИНИМАЛЬНАЯ НАСТРОЙКА

### Если есть OpenAI ключ:
```bash
export CUSTOM_LLM_URL=https://api.openai.com/v1/chat/completions
export OPENAI_API_KEY=sk-...
```

### Если есть Gemini:
```bash
export CUSTOM_LLM_URL=https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent
export GEMINI_API_KEY=...
```

### Если есть локальный Ollama:
```bash
export LLM_BACKENDS=ollama
export OLLAMA_HOST=http://localhost:11434
```

## ❓ ЧАСТЫЕ ВОПРОСЫ

**Q: Нужен ли MCP клиент?**
A: Нет! HTTP API работает с любым языком/инструментом.

**Q: Нужно ли настраивать схемы?**
A: Нет! Базовые схемы уже встроены.

**Q: Работает ли без GPU?**
A: Да! Используется внешний LLM API.

**Q: Можно ли в production?**
A: Для прода добавьте nginx и systemd (см. примеры ниже).

## 📱 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

### В Python:
```python
import requests

response = requests.post('http://localhost:8080/v1/apply-sgr', 
    json={'task': 'Проанализировать код', 'schema_type': 'analysis'})
print(response.json())
```

### В Node.js:
```javascript
fetch('http://localhost:8080/v1/apply-sgr', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task: 'Проанализировать код'})
}).then(r => r.json()).then(console.log)
```

### В n8n:
Просто добавьте HTTP Request ноду с URL: `http://your-server:8080/v1/apply-sgr`

## 🚨 ПРОБЛЕМЫ?

### "Module not found"
```bash
pip install -r requirements.txt
# или
pip install fastapi uvicorn httpx aiohttp pydantic
```

### "Connection refused"  
```bash
# Проверить что сервер запущен
ps aux | grep http_server
# Проверить порт
netstat -tlnp | grep 8080
```

### "No LLM configured"
```bash
# Добавить любой LLM
export CUSTOM_LLM_URL=https://api.openai.com/v1/chat/completions
export OPENAI_API_KEY=your-key
```

## 🎉 ВСЁ!

Серьезно, это всё что нужно для базового запуска. 
Остальные настройки - опциональны.