#!/bin/bash
# One-liner Docker deployment

docker run -d \
  --name mcp-sgr \
  -p 8080:8080 \
  -e CUSTOM_LLM_URL="${CUSTOM_LLM_URL:-https://api.openai.com/v1/chat/completions}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e HTTP_REQUIRE_AUTH=false \
  -e CACHE_ENABLED=true \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  mcp-sgr/mcp-sgr:latest

echo "✅ MCP-SGR запущен на http://localhost:8080"
echo "📊 Проверка: curl http://localhost:8080/health"
echo "🛑 Остановка: docker stop mcp-sgr"