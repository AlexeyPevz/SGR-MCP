#!/bin/bash
# 🚀 ПРОСТЕЙШЕЕ РАЗВЕРТЫВАНИЕ MCP-SGR ЗА 2 МИНУТЫ

echo "🚀 MCP-SGR Quick Deploy"
echo "======================"

# 1. Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите: apt install python3 python3-pip"
    exit 1
fi

# 2. Установка в один клик
echo "📦 Устанавливаю зависимости..."
pip3 install fastapi uvicorn httpx aiohttp pydantic jsonschema tenacity python-dotenv

# 3. Создание минимального .env
echo "⚙️ Создаю конфигурацию..."
cat > .env << EOF
# Минимальная конфигурация
LLM_BACKENDS=custom
CUSTOM_LLM_URL=https://api.openai.com/v1/chat/completions
HTTP_PORT=8080
CACHE_ENABLED=false
LOG_LEVEL=INFO
EOF

# 4. Создание директорий
mkdir -p data logs

# 5. Запуск
echo "✅ Готово! Запускаю сервер..."
echo ""
echo "Сервер будет доступен на: http://localhost:8080"
echo "Остановить: Ctrl+C"
echo ""

# Запуск HTTP сервера напрямую
python3 -m src.http_server