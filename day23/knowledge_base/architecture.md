# Архитектура AI Агентов

## Общая структура агента

Все наши агенты следуют единой архитектуре:

```
User Query
    ↓
[Main Agent]
    ↓
┌───────────────┐
│ RAG System    │ ← Поиск в документации
│ MCP Tools     │ ← Внешние данные/действия
│ Claude API    │ ← Генерация ответов
└───────────────┘
    ↓
Response
```

## Компоненты системы

### 1. RAG System (rag_system.py)

**Назначение:** Поиск релевантной информации в документах

**Как работает:**
```python
# Шаг 1: Создаем embeddings для документов
documents = ["doc1.md", "doc2.md"]
embeddings = ollama.embed(documents)  # Векторы

# Шаг 2: По запросу ищем похожие
query = "How to setup API?"
query_embedding = ollama.embed(query)
similar_docs = cosine_similarity(query_embedding, embeddings)

# Шаг 3: Отдаем топ результаты
top_3_docs = similar_docs[:3]
```

**Технические детали:**
- Модель: `nomic-embed-text` через Ollama
- Метрика: cosine similarity
- Порог релевантности: > 0.3
- Хранение: in-memory (список документов + embeddings)

### 2. MCP Server

**Назначение:** Предоставление инструментов для доступа к внешним данным

**Пример структуры:**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-server")

@app.tool()
def get_user_data(user_id: str):
    """Получить данные пользователя из CRM"""
    # Читаем из базы/файла
    return user_data

@app.tool()
def create_task(title: str, priority: str):
    """Создать новую задачу"""
    # Записываем в базу/файл
    return task_id
```

**Коммуникация:**
- Протокол: JSON-RPC через stdio
- Вход: stdin (запросы от клиента)
- Выход: stdout (ответы, JSON)
- Логи: stderr (НЕ stdout!)

### 3. Main Agent (main.py)

**Назначение:** Координация RAG и MCP для ответа на запросы

**Agentic Loop:**
```python
while True:
    # 1. Получаем запрос
    user_query = input()
    
    # 2. RAG поиск
    docs = rag.search(user_query)
    
    # 3. Формируем context для Claude
    context = f"Документация: {docs}"
    
    # 4. Вызываем Claude с MCP tools
    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": user_query}],
        tools=mcp_tools,  # MCP инструменты
        system=context    # RAG результаты
    )
    
    # 5. Обрабатываем tool calls
    if response.stop_reason == "tool_use":
        # Claude решил использовать MCP tool
        tool_result = execute_tool(response.content)
        # Отправляем результат обратно Claude
        continue
    
    # 6. Финальный ответ
    print(response.content)
```

## Паттерны использования

### RAG only (простые вопросы)
```
User: "Как работает RAG?"
→ RAG находит в README.md объяснение
→ Claude формулирует ответ
→ MCP не используется
```

### MCP only (действия)
```
User: "Создай задачу: fix bug"
→ RAG не находит релевантного
→ Claude вызывает MCP tool create_task()
→ Возвращает подтверждение
```

### RAG + MCP (комплексные запросы)
```
User: "Покажи high priority задачи и объясни, что такое MCP"
→ RAG находит объяснение MCP в architecture.md
→ Claude вызывает MCP tool get_tasks(priority="high")
→ Синтезирует ответ из обоих источников
```

## Обработка ошибок

### Типичные проблемы:

1. **Ollama не отвечает**
   - Причина: сервис не запущен
   - Решение: `ollama serve`

2. **MCP tool не найден**
   - Причина: неправильное имя в tool call
   - Решение: проверить @app.tool() названия

3. **API rate limit**
   - Причина: слишком много запросов
   - Решение: добавить time.sleep() между вызовами

4. **Embeddings timeout**
   - Причина: документ слишком большой
   - Решение: разбить на chunks по 500 слов

## Best Practices

1. **Всегда проверяй релевантность RAG**
   - Если score < 0.3, не используй документ
   - Лучше сказать "не знаю", чем галлюцинировать

2. **Логируй MCP вызовы**
   - Помогает дебажить
   - Используй stderr, не stdout

3. **Храни состояние правильно**
   - Для задач: JSON файл
   - Для embeddings: можно кэшировать в pickle

4. **Обрабатывай ошибки**
   - try-catch вокруг всех внешних вызовов
   - Понятные сообщения пользователю
