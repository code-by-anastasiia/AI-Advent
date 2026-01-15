# AI Agents Course - Project Documentation

## Обзор курса

Курс по разработке AI агентов с использованием Claude API, охватывающий темы от базового использования API до сложных multi-agent систем и Model Context Protocol (MCP).

## Цели обучения

- Понимание работы с Claude API
- Построение систем с RAG (Retrieval-Augmented Generation)
- Интеграция внешних сервисов через MCP
- Создание автономных AI агентов
- Комбинирование различных AI capabilities

## Технологический стек

- **Python 3.x** - основной язык
- **Anthropic Claude API** - AI модель
- **Ollama** - локальные embeddings (nomic-embed-text)
- **MCP (Model Context Protocol)** - интеграция внешних данных
- **Streamlit** - UI для некоторых проектов

## Структура проекта

```
ai-agents-course/
├── day20_developer_assistant/  # RAG + git integration
├── day21_code_review/          # RAG + code analysis
├── day22_support_assistant/    # RAG + MCP CRM
├── day23_team_assistant/       # RAG + MCP tasks (текущий)
└── shared/                     # Общие утилиты
    ├── rag_utils.py
    └── mcp_utils.py
```

## Ключевые концепции

### RAG (Retrieval-Augmented Generation)
Техника, позволяющая AI использовать внешние документы:
1. Превращаем документы в векторные embeddings
2. По запросу ищем похожие документы
3. Передаем найденное в Claude для генерации ответа

### MCP (Model Context Protocol)
Стандарт для подключения AI к внешним инструментам:
- Чтение данных (CRM, базы данных, API)
- Выполнение действий (создание задач, отправка сообщений)
- JSON-RPC коммуникация между клиентом и сервером

### Agentic Loop
Паттерн работы агента:
1. Получить запрос пользователя
2. Решить, какие инструменты нужны
3. Вызвать инструменты (RAG, MCP)
4. Синтезировать ответ
5. Повторить при необходимости

## Текущий прогресс

- ✅ День 20: Developer Assistant (RAG + git)
- ✅ День 21: Code Review (RAG + standards)
- ✅ День 22: Support Assistant (RAG + MCP)
- 🔄 День 23: Team Assistant (в процессе)

## Следующие шаги

1. Завершить День 23
2. Рефакторинг и улучшение качества кода
3. Добавление тестов
4. Унификация RAG систем
5. Улучшение обработки ошибок