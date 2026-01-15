# Завершенные проекты

## День 20: Developer Assistant

**Дата:** 12.01.2026

**Задача:** Создать ассистента для разработчика, который знает проект и может отвечать на вопросы по коду.

**Реализация:**
- ProjectRAG класс для индексации README.md и Python файлов
- Генерация embeddings через Ollama (nomic-embed-text)
- Поиск по similarity для нахождения релевантных документов
- Streamlit интерфейс для взаимодействия
- Интеграция с git (изначально планировалась, упрощена для Windows)

**Технологии:**
- Python 3.x
- Ollama для embeddings
- Anthropic Claude API
- Streamlit для UI

**Чему научилась:**
- Как работают vector embeddings
- Cosine similarity для поиска
- Разница между index и search фазами RAG
- Windows-specific проблемы с git integration

**Трудности:**
- Установка и настройка Ollama
- Понимание, зачем нужны embeddings (почему не просто text search)
- Git integration на Windows

**Результат:**
✅ Работающий ассистент, который отвечает на вопросы по документации проекта

---

## День 21: Code Review Automation

**Дата:** 13.01.2026

**Задача:** Автоматизировать review кода с использованием RAG для стандартов кода.

**Реализация:**
- RAG система с документами coding standards (security, style, best practices)
- Mock PR data (old_code.py, new_code.py) для демонстрации
- Semantic search для нахождения релевантных стандартов
- AI анализ изменений с конкретными рекомендациями
- Structured output: security score, code quality, recommendations

**Технологии:**
- Python, Ollama, Claude API
- RAG для поиска стандартов
- Прямое чтение файлов (без GitHub integration)

**Чему научилась:**
- RAG можно использовать не только для FAQ, но и для стандартов/правил
- Как структурировать AI output для code review
- Важность mock data для тестирования без реальных зависимостей

**Трудности:**
- Первоначальная версия была слишком сложной (GitHub API, MCP)
- Пришлось упростить до local-only версии
- Установка зависимостей (bcrypt, python-dotenv)
- MCP API changes - пришлось убрать MCP integration

**Результат:**
✅ Система анализирует код и дает рекомендации на основе coding standards

---

## День 22: Support Assistant

**Дата:** 14.01.2026

**Задача:** Создать ассистента поддержки, который использует FAQ и персональные данные пользователей.

**Реализация:**
- RAG для knowledge base (FAQ, sync guides, troubleshooting)
- MCP server для доступа к CRM данным
- Интерактивный CLI режим с командами `/user <id> <вопрос>`
- Детальное логирование для демонстрации работы системы
- Три типа ответов:
  1. Только документация (high RAG score)
  2. Общие знания Claude (low RAG score)
  3. Персонализация (RAG + MCP data)

**Технологии:**
- Python, Ollama, Claude API
- MCP Python library для server
- JSON для хранения CRM данных

**Чему научилась:**
- Как комбинировать RAG и MCP в одном агенте
- Когда использовать какой источник данных
- Importance of logging для понимания decision-making процесса
- MCP stdout должен быть чистым (JSON-RPC only, дебаг в stderr)

**Трудности:**
- Понимание, когда вызывать MCP tools vs использовать только RAG
- Authentication errors с Anthropic API (неправильный API key в .env)
- Model naming - использовала старое имя модели
- Создание интерактивного режима для удобного тестирования

**Результат:**
✅ Работающий support assistant с RAG + MCP integration

**Ключевые инсайты:**
- RAG для общей информации (документация)
- MCP для персональных данных (user-specific)
- Claude решает, что использовать на основе запроса
- Важность прозрачности (логирование decisions)

---

## Общие выводы по Дням 20-22

### Прогресс в понимании:
1. **RAG** - от базового "что это" до уверенного использования
2. **MCP** - понимание протокола и best practices
3. **Комбинирование** - когда использовать что

### Технические навыки:
- Работа с Ollama embeddings
- MCP server development
- Claude API integration
- Error handling и debugging

### Что улучшить:
- Более чистый код (рефакторинг)
- Унификация RAG систем
- Добавление тестов
- Лучшая обработка ошибок