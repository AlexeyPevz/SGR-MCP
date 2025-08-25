# Технический долг MCP-SGR

## 🔴 Приоритет 1: Критические долги

### 1. Доработать learn_schema_from_examples
**Статус**: Базовая реализация есть, нужна полная
**Что сделать**:
- [ ] Реализовать анализ паттернов из примеров
- [ ] Автоматическая генерация JSON Schema
- [ ] Валидация выученных схем
- [ ] Сохранение в catalog

### 2. Реализовать интеграции
**Статус**: Директория пустая
**Что сделать**:
- [ ] LangChain middleware (2-3 дня)
- [ ] AutoGen wrapper (1-2 дня)
- [ ] n8n custom node (1 день)
- [ ] CrewAI integration (2 дня)

### 3. Добавить метрики OpenTelemetry
**Статус**: TODO в коде
**Что сделать**:
- [ ] Histogram для latency
- [ ] Counter для вызовов
- [ ] Gauge для cache size
- [ ] Custom метрики confidence

## 🟡 Приоритет 2: Roadmap фичи

### 4. Schema Catalog
**Что сделать**:
- [ ] API для управления схемами
- [ ] Версионирование схем
- [ ] Import/Export
- [ ] Public registry интерфейс

### 5. A/B Testing Framework
**Что сделать**:
- [ ] Experiment runner
- [ ] Metrics collector
- [ ] Statistical analysis
- [ ] Reporting

## 🟢 Приоритет 3: Новые фичи

### 6. RAG Integration
**Что сделать**:
- [x] Базовые схемы (сделано!)
- [ ] Интеграция с vector stores
- [ ] Reranking pipeline
- [ ] Iterative retrieval

### 7. Advanced Confidence
**Что сделать**:
- [ ] Multi-factor confidence
- [ ] Cross-validation
- [ ] Consistency checks
- [ ] Confidence calibration

## План реализации

### Неделя 1: Критические долги
- День 1-2: learn_schema_from_examples
- День 3-4: n8n node + LangChain basics
- День 5: OpenTelemetry metrics

### Неделя 2: Интеграции
- День 1-2: LangChain полная интеграция
- День 3: AutoGen wrapper
- День 4-5: CrewAI + тесты

### Неделя 3: Новые фичи
- День 1-2: RAG полная интеграция
- День 3: Advanced confidence
- День 4-5: A/B testing basics

## Быстрые победы (можно сделать за 1-2 часа)

1. **Metrics stub → working metrics**:
```python
# В telemetry.py заменить TODO на:
self.meter = metrics.get_meter("mcp-sgr")
self.call_counter = self.meter.create_counter("sgr_calls_total")
self.latency_histogram = self.meter.create_histogram("sgr_latency_ms")
```

2. **n8n HTTP examples** (уже частично есть)
3. **LangChain basic wrapper**:
```python
from langchain.schema.runnable import Runnable
class SGRRunnable(Runnable):
    def invoke(self, input, config=None):
        return apply_sgr_tool({"task": input})
```

## Метрики успеха

- [ ] 100% заявленных инструментов работают
- [ ] 4+ интеграции реализованы
- [ ] Метрики собираются и экспортируются
- [ ] Примеры для каждой интеграции
- [ ] CI проходит для всех компонентов