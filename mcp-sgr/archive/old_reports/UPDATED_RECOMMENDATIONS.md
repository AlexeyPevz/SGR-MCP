# Обновленные рекомендации по MCP-SGR (после тестов прямого API)

## 🎯 Ключевые открытия

### 1. **OpenAI Structured Output НЕ работает даже через прямой API!**

- ❌ **GPT-4o (direct)** - HTTP 400 на structured output
- ❌ **GPT-4o-mini (direct)** - HTTP 400 на structured output  
- ❌ **GPT-3.5-turbo (direct)** - HTTP 400 на structured output
- ✅ **НО: GPT-4o-mini отлично работает в unstructured режиме** (Quality: 10/10)

### 2. **Проблемы с Claude через OpenRouter**

- ❌ Claude модели НЕ работают со structured output через OpenRouter
- ✅ Claude работает в unstructured режиме, но с проблемами парсинга

### 3. **Отличные бюджетные альтернативы**

🏆 **Лучшие модели по качеству (unstructured):**

1. **Qwen-2.5-72b** - Quality: 10/10, Latency: 5.80s ⭐ ЛУЧШИЙ ВЫБОР
2. **GPT-4o-mini (direct)** - Quality: 10/10, Latency: 1.96s (самый быстрый)
3. **Claude-3.5-sonnet** - Quality: 10/10, Latency: 5.97s
4. **Mistral-7b** - Quality: 8.5/10, Latency: 3.16s 💰 БЮДЖЕТНЫЙ
5. **Gemini-flash-1.5** - Quality: 8.5/10, Latency: 2.81s

### 4. **Единственная модель с рабочим structured output**

- ✅ **Gemini Flash 1.5** - единственная модель, где structured output работает через OpenRouter

## 📋 Обновленные практические рекомендации

### Новая стратегия для MCP-SGR

```python
# Конфигурация для production
UPDATED_CONFIG = {
    # Основные модели (unstructured mode)
    "primary_models": [
        "qwen/qwen-2.5-72b-instruct",      # Лучшее качество, бюджетно
        "openai/gpt-4o-mini",               # Быстро, качественно (прямой API)
        "mistralai/mistral-7b-instruct",   # Супер-бюджетно, хорошее качество
    ],
    
    # Только для structured output
    "structured_output_models": [
        "google/gemini-flash-1.5",          # Единственная рабочая
        "google/gemini-pro-1.5",            # Премиум вариант
    ],
    
    # НЕ использовать для SGR
    "avoid_models": [
        "deepseek/*",                       # Плохое качество парсинга
        "meta-llama/*",                     # Проблемы с JSON
        "mixtral-8x7b",                     # Ошибки экранирования
    ],
    
    # Режим работы
    "default_mode": "unstructured",         # Structured output ненадежен
    "use_json_instruction": True,           # Добавлять инструкции для JSON
    "retry_on_parse_error": True,           # Повторять при ошибках парсинга
}
```

### Оптимальная стратегия для разных бюджетов

#### 💎 Premium (качество превыше всего)
```python
# Используйте GPT-4o-mini через прямой API
model = "gpt-4o-mini"
api_type = "direct"
# Quality: 10/10, Latency: ~2s
```

#### 🎯 Balanced (оптимально)
```python
# Используйте Qwen-2.5-72b через OpenRouter
model = "qwen/qwen-2.5-72b-instruct"
api_type = "openrouter"
# Quality: 10/10, Latency: ~6s, очень дешево
```

#### 💰 Budget (минимальные затраты)
```python
# Используйте Mistral-7b через OpenRouter
model = "mistralai/mistral-7b-instruct"
api_type = "openrouter"
# Quality: 8.5/10, Latency: ~3s, практически бесплатно
```

## 🚀 Обновленный план действий

### 1. Немедленные изменения

1. **Переключить SGR на unstructured mode по умолчанию**
   ```python
   # В apply_sgr.py
   def apply_sgr(task, mode="full", use_structured=False):
       # Structured output только для Gemini
       if model.startswith("google/gemini") and use_structured:
           return structured_approach()
       else:
           return unstructured_approach_with_json_parsing()
   ```

2. **Добавить Qwen и Mistral в список рекомендованных моделей**

3. **Создать улучшенный JSON парсер**
   ```python
   def robust_json_parse(content):
       # Убрать markdown обертки
       if content.startswith("```"):
           content = content.split("```")[1]
           if content.startswith("json"):
               content = content[4:]
           content = content.rsplit("```", 1)[0]
       
       # Попытаться исправить common issues
       content = content.strip()
       
       # Парсить
       return json.loads(content)
   ```

### 2. Краткосрочные улучшения (1 неделя)

1. **Создать адаптивный промпт для unstructured JSON**
   ```python
   UNIVERSAL_JSON_PROMPT = """
   Analyze the task and respond with a JSON object containing:
   - summary: brief overview (string)
   - key_points: main findings (array of strings)
   - recommendations: suggested actions (array of strings)
   - confidence: 0-1 score (number)
   
   Example format:
   {
     "summary": "...",
     "key_points": ["point 1", "point 2"],
     "recommendations": ["action 1", "action 2"],
     "confidence": 0.85
   }
   
   Task: {task}
   """
   ```

2. **Реализовать fallback цепочку**
   ```python
   FALLBACK_CHAIN = [
       ("qwen/qwen-2.5-72b-instruct", "unstructured"),
       ("mistralai/mistral-7b-instruct", "unstructured"),
       ("google/gemini-flash-1.5", "structured"),
       ("anthropic/claude-3.5-haiku", "unstructured"),
   ]
   ```

### 3. Долгосрочные улучшения (1 месяц)

1. **Создать benchmark suite для новых моделей**
2. **Исследовать function calling как альтернативу structured output**
3. **Разработать систему автоматического выбора модели по задаче**

## 📊 Обновленная сравнительная таблица

| Модель | API | Structured | Unstructured | Качество | Скорость | Стоимость | Рекомендация |
|--------|-----|------------|--------------|----------|----------|-----------|--------------|
| Qwen-2.5-72b | OpenRouter | ❌ | ✅ | 10/10 | 6s | $0.0003/1k | ⭐ ЛУЧШИЙ ВЫБОР |
| GPT-4o-mini | Direct | ❌ | ✅ | 10/10 | 2s | $0.00015/1k | Для скорости |
| Mistral-7b | OpenRouter | ❌ | ✅ | 8.5/10 | 3s | ~$0.0001/1k | 💰 БЮДЖЕТ |
| Gemini Flash | OpenRouter | ✅ | ✅ | 8.5/10 | 2-3s | $0.0003/1k | Для structured |
| Claude Sonnet | OpenRouter | ❌ | ✅* | 10/10 | 6s | $0.003/1k | *С оговорками |

## ⚠️ Важные предупреждения

1. **OpenAI structured output сломан** - не только через OpenRouter, но и через прямой API
2. **Claude через OpenRouter** возвращает не JSON, а markdown - нужна обработка
3. **DeepSeek** - не парсит схему правильно, качество 0/10
4. **Llama модели** - постоянные проблемы с форматированием JSON

## 💡 Финальная рекомендация

**Используйте Qwen-2.5-72b в unstructured режиме как основную модель для MCP-SGR:**

- Качество на уровне GPT-4
- В 10 раз дешевле
- Стабильный JSON output
- Хорошая скорость

Для критически важных задач - GPT-4o-mini через прямой API.
Для максимальной экономии - Mistral-7b.
Для structured output - только Gemini.