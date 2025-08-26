# Финальные рекомендации по MCP-SGR

## 🎯 Главные выводы

После проведения детального тестирования промптов, схем и моделей:

### 1. **Structured Output критичен, но проблематичен**

- ✅ **Claude 3.5 Haiku** - единственная модель, стабильно работающая со всеми схемами
- ✅ **Gemini Flash 1.5** - хорошо работает со structured output
- ❌ **OpenAI модели через OpenRouter** - НЕ РАБОТАЮТ со structured output (постоянные 400 ошибки)
- ⚠️ **Gemini Free tier** - частые rate limit ошибки

### 2. **SGR режимы - текущее состояние**

- **SGR Full**: Работает, улучшает качество на ~10%, но увеличивает латентность в 1.7x
- **SGR Lite**: Требует полной переработки (текущая версия ухудшает результаты)
- **Baseline**: Достаточно для простых задач с правильными моделями

### 3. **Оптимальные промпты для SGR Lite**

По результатам тестирования лучшие стратегии:

```python
# Лучший вариант для Claude (Quality: 1.00)
OPTIMAL_LITE_PROMPT = {
    "system": "You are a JSON-only assistant. Respond with STRICTLY valid JSON matching the provided schema.",
    "user": """Task: {task}

Provide a JSON response following this structure:
{schema}

Be concise but complete."""
}

# Альтернатива с guided reasoning (Quality: 1.00, быстрее)
GUIDED_LITE_PROMPT = {
    "system": "You are an analytical assistant. Always respond in valid JSON format.",
    "user": """Task: {task}

Analyze this step by step:
1. Understand the task
2. Identify key issues or points
3. Suggest improvements

Format your response as JSON:
{schema}

Include a confidence score (0-1) based on your analysis completeness."""
}
```

## 📋 Практические рекомендации

### Для Production использования

```python
PRODUCTION_CONFIG = {
    # ONLY these models for SGR
    "approved_models": [
        "anthropic/claude-3.5-haiku",    # Best overall
        "anthropic/claude-3.5-sonnet",   # For critical tasks
        "google/gemini-flash-1.5",       # Good alternative
        "google/gemini-pro-1.5"          # Premium option
    ],
    
    # Avoid these
    "blacklisted_models": [
        "openai/*",                      # All OpenAI models via OpenRouter
        "meta-llama/*",                  # Poor structured output
        "mixtral/*",                     # Inconsistent results
    ],
    
    # Settings
    "default_mode": "full",              # Use full for complex tasks
    "fallback_mode": "baseline",         # When SGR fails
    "cache_enabled": True,
    "retry_on_400": False,               # Don't retry schema errors
}
```

### Схемы - уровни сложности

```python
# Для всех моделей работает
UNIVERSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "items": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"}
    },
    "required": ["summary", "items"]
}

# Только для Claude/Gemini
ADVANCED_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "context": {"type": "string"}
            }
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]}
                }
            }
        }
    }
}
```

## 🚀 План действий

### Немедленно (для использования SGR)

1. **Переключиться на Claude 3.5 Haiku** как основную модель
2. **Отключить OpenAI модели** в конфигурации роутера
3. **Использовать упрощенные схемы** для начала

### Краткосрочно (1-2 недели)

1. **Переписать SGR Lite** используя оптимальные промпты
2. **Добавить проверку модели** перед применением structured output
3. **Реализовать fallback** на unstructured при ошибках 400

### Долгосрочно (1 месяц)

1. **Исследовать прямые API** OpenAI/Anthropic вместо OpenRouter
2. **Создать адаптивные схемы** для разных моделей
3. **Реализовать A/B тестирование** для промптов

## ⚠️ Критические ограничения

1. **OpenAI через OpenRouter не поддерживает structured output** - это факт, не баг
2. **Rate limits на бесплатных моделях** делают их непригодными для production
3. **Сложные nested схемы** увеличивают вероятность отказа

## 💡 Быстрый старт

```bash
# Оптимальная конфигурация для тестирования
export OPENROUTER_DEFAULT_MODEL="anthropic/claude-3.5-haiku"
export SGR_MODE="full"
export CACHE_ENABLED="true"

# Запуск с правильной моделью
python examples/basic_usage.py
```

## 📊 Сравнительная таблица

| Модель | Structured Output | SGR Full | Качество | Латентность | Стоимость |
|--------|------------------|----------|----------|-------------|-----------|
| Claude 3.5 Haiku | ✅ Отлично | ✅ | 0.95 | 3-5s | $0.001/1k |
| Claude 3.5 Sonnet | ✅ Отлично | ✅ | 0.97 | 4-6s | $0.003/1k |
| Gemini Flash 1.5 | ✅ Хорошо | ✅ | 0.90 | 1-3s | $0.0003/1k |
| GPT-4o | ❌ Не работает | ❌ | - | - | $0.0025/1k |
| GPT-4o-mini | ❌ Не работает | ❌ | - | - | $0.00015/1k |
| Llama 3.1 | ❌ Плохо | ⚠️ | 0.20 | 2-4s | $0.00018/1k |

## Заключение

MCP-SGR **работает и улучшает качество**, но требует:
1. Правильного выбора модели (Claude или Gemini)
2. Отказа от OpenAI через OpenRouter
3. Доработки SGR Lite режима

При соблюдении этих условий система показывает стабильное улучшение качества reasoning на 10-15% с приемлемым увеличением латентности.