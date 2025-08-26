#!/usr/bin/env python3
"""
Адаптер для работы с YandexGPT API в системе бенчмарков
"""

import json
import urllib.request
import urllib.error
import os
import time
from typing import Dict, List, Optional

class YandexGPTAdapter:
    """Адаптер для работы с YandexGPT API"""
    
    def __init__(self, api_key: Optional[str] = None, folder_id: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
    def call_model(self, 
                   messages: List[Dict[str, str]], 
                   model: str = "yandexgpt-lite",
                   temperature: float = 0.6,
                   max_tokens: int = 1000,
                   sgr_mode: Optional[str] = None) -> Dict:
        """
        Вызов YandexGPT API
        
        Args:
            messages: Список сообщений чата
            model: Модель (yandexgpt-lite или yandexgpt)
            temperature: Температура генерации
            max_tokens: Максимум токенов
            sgr_mode: Режим SGR (для добавления схемы)
            
        Returns:
            Dict с результатом
        """
        
        if not self.api_key or not self.folder_id:
            return {
                "success": False,
                "error": "Missing YANDEX_API_KEY or YANDEX_FOLDER_ID"
            }
        
        # Подготовка промпта для YandexGPT
        prompt = self._prepare_prompt(messages, sgr_mode)
        
        # Формирование запроса
        model_uri = f"gpt://{self.folder_id}/{model}/latest"
        
        request_data = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        # Создание HTTP запроса
        req = urllib.request.Request(
            self.base_url,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
                "x-folder-id": self.folder_id
            },
            data=json.dumps(request_data).encode('utf-8')
        )
        
        try:
            start_time = time.time()
            response = urllib.request.urlopen(req)
            result = json.loads(response.read().decode('utf-8'))
            latency = time.time() - start_time
            
            # Извлечение ответа
            if "result" in result and "alternatives" in result["result"]:
                content = result["result"]["alternatives"][0]["message"]["text"]
                tokens_used = result["result"]["usage"]["totalTokens"]
                
                return {
                    "success": True,
                    "content": content,
                    "model": model,
                    "latency": latency,
                    "tokens_used": tokens_used,
                    "cost": self._calculate_cost(tokens_used, model)
                }
            else:
                return {
                    "success": False,
                    "error": "Unexpected response format"
                }
                
        except urllib.error.HTTPError as e:
            error_data = e.read().decode('utf-8')
            return {
                "success": False,
                "error": f"HTTP {e.code}: {error_data}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_prompt(self, messages: List[Dict[str, str]], sgr_mode: Optional[str]) -> str:
        """Подготовка промпта с учетом SGR"""
        
        # Извлекаем последнее сообщение пользователя
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Добавляем SGR схему если нужно
        if sgr_mode == "lite":
            prompt = f"""{user_message}

Пожалуйста, структурируйте ваш ответ в формате JSON:
{{
  "task_understanding": "что нужно сделать",
  "solution": "решение задачи"
}}"""
        elif sgr_mode == "full":
            prompt = f"""{user_message}

Пожалуйста, структурируйте ваш ответ в формате JSON:
{{
  "requirements_analysis": "анализ требований",
  "approach": "подход к решению",
  "implementation": "реализация",
  "validation": "проверка решения"
}}"""
        else:
            prompt = user_message
        
        return prompt
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Расчет стоимости запроса"""
        
        # Примерные цены в долларах
        costs = {
            "yandexgpt-lite": 0.002,  # $0.002 за 1k токенов
            "yandexgpt": 0.02         # $0.02 за 1k токенов
        }
        
        cost_per_1k = costs.get(model, 0.002)
        return (tokens / 1000) * cost_per_1k


def test_yandex_gpt():
    """Простой тест YandexGPT"""
    
    print("🧪 Тестирование YandexGPT...")
    
    adapter = YandexGPTAdapter()
    
    # Тест без SGR
    print("\n1️⃣ Тест БЕЗ SGR:")
    result = adapter.call_model(
        messages=[{"role": "user", "content": "Напиши функцию на Python для проверки четности числа"}],
        model="yandexgpt-lite",
        sgr_mode=None
    )
    
    if result["success"]:
        print(f"✅ Успешно! Латентность: {result['latency']:.2f}s")
        print(f"Ответ: {result['content'][:200]}...")
    else:
        print(f"❌ Ошибка: {result['error']}")
    
    # Тест с SGR-lite
    print("\n2️⃣ Тест С SGR-lite:")
    result = adapter.call_model(
        messages=[{"role": "user", "content": "Напиши функцию на Python для проверки четности числа"}],
        model="yandexgpt-lite",
        sgr_mode="lite"
    )
    
    if result["success"]:
        print(f"✅ Успешно! Латентность: {result['latency']:.2f}s")
        print(f"Структурированный ответ: {result['content'][:300]}...")
    else:
        print(f"❌ Ошибка: {result['error']}")


if __name__ == "__main__":
    test_yandex_gpt()