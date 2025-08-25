#!/bin/bash
# Минимальный тест работоспособности

echo "🧪 Тестирование MCP-SGR API..."
echo

# 1. Health check
echo "1️⃣ Проверка здоровья:"
curl -s http://localhost:8080/health | python3 -m json.tool

# 2. Простой анализ
echo -e "\n2️⃣ Тест анализа:"
curl -s -X POST http://localhost:8080/v1/apply-sgr \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Как оптимизировать запрос SELECT * FROM users WHERE status = active",
    "schema_type": "analysis",
    "budget": "lite"
  }' | python3 -m json.tool | head -20

# 3. Улучшение промпта
echo -e "\n3️⃣ Тест улучшения промпта:"
curl -s -X POST http://localhost:8080/v1/enhance-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "original_prompt": "напиши функцию сортировки"
  }' | python3 -m json.tool | grep -A5 "enhanced_prompt"

echo -e "\n✅ Если видите JSON ответы - всё работает!"