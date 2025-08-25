# Подключение MCP-SGR к Claude Desktop

## Способ 1: Локальный запуск с Gemini

### 1. Настройка Gemini прокси

Создайте файл `run-with-gemini.sh`:

```bash
#!/bin/bash
export GEMINI_API_KEY="your-gemini-api-key"

# Запустить Gemini прокси в фоне
python examples/gemini_proxy.py &
PROXY_PID=$!

# Запустить MCP сервер
python -m src.server

# Остановить прокси при завершении
kill $PROXY_PID
```

### 2. Конфигурация Claude Desktop

Добавьте в `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) или аналогичный файл на других платформах:

```json
{
  "mcpServers": {
    "sgr-reasoning": {
      "command": "/bin/bash",
      "args": ["/path/to/mcp-sgr/run-with-gemini.sh"],
      "env": {
        "GEMINI_API_KEY": "your-gemini-api-key",
        "LLM_BACKENDS": "custom",
        "CUSTOM_LLM_URL": "http://localhost:8001/v1/chat/completions"
      }
    }
  }
}
```

## Способ 2: Подключение к удаленному VPS

### 1. На VPS запустите HTTP сервер

```bash
cd mcp-sgr
export GEMINI_API_KEY="your-key"

# Запустить в screen/tmux
screen -S mcp-sgr
python examples/gemini_proxy.py &
python -m src.http_server
# Ctrl+A, D для выхода из screen
```

### 2. На локальной машине настройте мост

```bash
# Установите Node.js если нет
npm install

# Сделайте мост исполняемым
chmod +x examples/mcp-http-bridge.js
```

### 3. Конфигурация Claude Desktop для VPS

```json
{
  "mcpServers": {
    "sgr-reasoning-vps": {
      "command": "node",
      "args": ["/path/to/mcp-sgr/examples/mcp-http-bridge.js"],
      "env": {
        "SGR_HTTP_URL": "http://your-vps-ip:8080",
        "SGR_API_KEY": "optional-api-key"
      }
    }
  }
}
```

## Способ 3: Использование с Ollama (локально)

```json
{
  "mcpServers": {
    "sgr-reasoning-ollama": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/mcp-sgr",
      "env": {
        "PYTHONPATH": "/path/to/mcp-sgr",
        "LLM_BACKENDS": "ollama",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_DEFAULT_MODEL": "llama3.1:8b"
      }
    }
  }
}
```

## Использование в Claude Desktop

После добавления конфигурации и перезапуска Claude Desktop, вы сможете использовать команды:

1. **Анализ задачи**:
   ```
   Используй apply_sgr чтобы проанализировать архитектуру микросервисов для e-commerce платформы
   ```

2. **Улучшение промпта**:
   ```
   Используй enhance_prompt_with_sgr чтобы улучшить промпт "напиши парсер логов"
   ```

3. **Обертка агента** (если есть другой агент):
   ```
   Используй wrap_agent_call для анализа ответа от coding агента
   ```

## Проверка подключения

В Claude Desktop должны появиться инструменты:
- `apply_sgr` - применить схему reasoning
- `enhance_prompt_with_sgr` - улучшить промпт
- `wrap_agent_call` - обернуть вызов агента
- `learn_schema_from_examples` - обучить новую схему

## Troubleshooting

1. **Сервер не запускается**:
   - Проверьте что установлены все зависимости: `pip install -e .`
   - Проверьте логи в `logs/mcp-sgr.log`

2. **Claude не видит инструменты**:
   - Перезапустите Claude Desktop полностью
   - Проверьте путь к конфигурации
   - Посмотрите консоль разработчика (Cmd+Opt+I на macOS)

3. **Ошибки при вызове**:
   - Проверьте что Gemini API ключ правильный
   - Убедитесь что прокси запущен (порт 8001)
   - Проверьте доступность VPS если используете удаленный сервер

## Безопасность

- Для VPS используйте HTTPS через nginx
- Установите `HTTP_AUTH_TOKEN` в `.env`
- Ограничьте доступ по IP через firewall
- Не храните API ключи в конфигурации, используйте переменные окружения