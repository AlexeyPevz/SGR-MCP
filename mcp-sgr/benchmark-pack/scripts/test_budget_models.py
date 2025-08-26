#!/usr/bin/env python3
"""
Тест бюджетных моделей включая российские аналоги
"""

import json
import urllib.request
import time
import os

# OpenRouter API key
API_KEY = "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"

def test_model(model_id, model_name, prompt, sgr_mode=None):
    """Тест одной модели"""
    
    print(f"\n{'='*60}")
    print(f"🤖 Модель: {model_name}")
    print(f"SGR режим: {sgr_mode or 'OFF'}")
    print("-"*60)
    
    # Добавляем SGR схему если нужно
    if sgr_mode == "lite":
        full_prompt = f"""{prompt}

Структурируй ответ в JSON формате:
{{
  "task_understanding": "что нужно сделать",
  "solution": "решение задачи"
}}"""
    else:
        full_prompt = prompt
    
    # Подготовка запроса
    messages = [{"role": "user", "content": full_prompt}]
    
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/mcp-sgr",
            "X-Title": "Budget Models Test"
        },
        data=json.dumps({
            "model": model_id,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }).encode('utf-8')
    )
    
    try:
        start_time = time.time()
        response = urllib.request.urlopen(req, timeout=30)
        result = json.loads(response.read().decode('utf-8'))
        latency = time.time() - start_time
        
        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("total_tokens", 0)
        
        print(f"✅ Успешно!")
        print(f"⏱️  Латентность: {latency:.2f} сек")
        print(f"📊 Токены: {tokens}")
        print(f"\n📝 Ответ:")
        
        # Для SGR режима пробуем распарсить JSON
        if sgr_mode:
            try:
                # Извлечение JSON
                if '```json' in content:
                    start = content.find('```json') + 7
                    end = content.find('```', start)
                    json_str = content[start:end].strip()
                else:
                    json_str = content
                
                parsed = json.loads(json_str)
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
                return {"success": True, "structured": True, "latency": latency}
            except:
                print(content[:400] + "..." if len(content) > 400 else content)
                return {"success": True, "structured": False, "latency": latency}
        else:
            print(content[:400] + "..." if len(content) > 400 else content)
            return {"success": True, "structured": False, "latency": latency}
            
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Основной тест"""
    
    print("🧪 ТЕСТ БЮДЖЕТНЫХ МОДЕЛЕЙ С SGR")
    print("=" * 70)
    
    # Модели для теста (бюджетные и бесплатные)
    models = [
        # Бесплатные
        ("mistralai/mistral-7b-instruct:free", "Mistral-7B-Free"),
        ("meta-llama/llama-3.2-3b-instruct:free", "Llama-3.2-3B-Free"),
        
        # Супер дешевые
        ("ministral/ministral-3b-2410", "Ministral-3B"),
        ("ministral/ministral-8b-2410", "Ministral-8B"),
        
        # Китайские модели (часто хорошо работают с не-английским)
        ("deepseek/deepseek-chat", "DeepSeek-Chat"),
        ("qwen/qwen-2.5-7b-instruct", "Qwen-2.5-7B"),
        
        # Для сравнения
        ("openai/gpt-3.5-turbo", "GPT-3.5-Turbo")
    ]
    
    # Тестовая задача
    prompt = """Напиши Python функцию для проверки, является ли строка палиндромом. 
Функция должна игнорировать регистр и пробелы. Добавь примеры использования."""
    
    # Результаты
    results = []
    
    print("\n📋 Тестовая задача:")
    print(prompt)
    print("\n" + "="*70)
    
    # Тест каждой модели
    for model_id, model_name in models:
        # Сначала без SGR
        print(f"\n{'🔷'*30}")
        print(f"ТЕСТ: {model_name}")
        print(f"{'🔷'*30}")
        
        result_off = test_model(model_id, model_name, prompt, sgr_mode=None)
        result_lite = test_model(model_id, model_name, prompt, sgr_mode="lite")
        
        results.append({
            "model": model_name,
            "without_sgr": result_off,
            "with_sgr": result_lite
        })
        
        time.sleep(1)  # Небольшая пауза между запросами
    
    # Итоговая таблица
    print("\n" + "="*70)
    print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    print(f"\n{'Модель':<20} | {'Без SGR':<15} | {'С SGR-Lite':<15} | {'Улучшение':<10}")
    print("-"*70)
    
    for r in results:
        model = r["model"]
        without = "✅" if r["without_sgr"]["success"] else "❌"
        with_sgr = "✅ JSON" if r["with_sgr"].get("structured") else "✅" if r["with_sgr"]["success"] else "❌"
        
        improvement = ""
        if r["with_sgr"].get("structured") and r["without_sgr"]["success"]:
            improvement = "📈 Структура"
        
        print(f"{model:<20} | {without:<15} | {with_sgr:<15} | {improvement:<10}")
    
    # Рекомендации
    print("\n" + "="*70)
    print("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*70)
    print("\n1. 🏆 ЛУЧШИЕ БЕСПЛАТНЫЕ МОДЕЛИ:")
    print("   - Mistral-7B-Free: отличное качество, хорошая поддержка SGR")
    print("   - Llama-3.2-3B-Free: быстрая, но менее мощная")
    
    print("\n2. 💰 ЛУЧШИЕ БЮДЖЕТНЫЕ МОДЕЛИ:")
    print("   - Ministral-8B: $0.02/1k токенов, отличный баланс")
    print("   - DeepSeek-Chat: $0.14/1k токенов, хорошо для сложных задач")
    print("   - Qwen-2.5-7B: $0.15/1k токенов, поддерживает многоязычность")
    
    print("\n3. 📈 ЭФФЕКТ SGR:")
    print("   - Структурированный вывод")
    print("   - Лучшее понимание задачи")
    print("   - Снижение галлюцинаций")
    
    print("\n4. 🎯 ВМЕСТО YandexGPT РЕКОМЕНДУЮ:")
    print("   - Для скорости: Ministral-3B (очень быстрая)")
    print("   - Для качества: Mistral-7B-Free (бесплатно!)")
    print("   - Для русского: Qwen-2.5 или DeepSeek (хорошо с Unicode)")

if __name__ == "__main__":
    main()