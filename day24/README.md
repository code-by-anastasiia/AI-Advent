# Автоматический генератор документации

Инструмент для автоматизации создания документации Python проектов с использованием Claude API. Анализирует структуру кода и генерирует профессиональные README.md файлы.

## Описание

Система автоматически создаёт документацию для Python проектов, анализируя код через AST и генерируя README.md через Claude API. Поддерживает как разовую генерацию, так и автоматическое отслеживание изменений с обновлением документации в реальном времени.

## Структура проекта

```
day_24/
├── code_parser.py          # Парсер Python кода - извлекает структуру проекта
├── demo_parse.py           # Демо версия без Claude API для тестирования
├── doc_generator.py        # Генератор документации через Claude API
├── main.py                 # Главный модуль с CLI интерфейсом
├── watch.py                # Автоматический режим отслеживания изменений
└── README.md              # Документация проекта
```

**Описание файлов:**
- `code_parser.py` - Парсит Python код, извлекает функции, классы, импорты
- `demo_parse.py` - Демонстрация парсинга проекта без API ключа
- `doc_generator.py` - Генерирует README.md на основе структуры проекта
- `main.py` - Автоматический генератор документации с CLI
- `watch.py` - Автоматическое отслеживание изменений в проектах

## Установка

```bash
# Установка зависимостей
pip install anthropic python-dotenv

# Настройка API ключа
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Использование

### Основные режимы работы

```bash
# Генерация документации для одного проекта
python main.py
# Выберите опцию 1, укажите путь к проекту

# Пакетная генерация для всех проектов в папке
python main.py
# Выберите опцию 2, укажите путь к папке с проектами

# Демо режим (без API ключа)
python demo_parse.py
```

### Автоматический режим

```bash
# Запуск отслеживания изменений
python watch.py
# Укажите папку для мониторинга
# При изменении .py файлов автоматически обновляет README
```

### Программное использование

```python
from code_parser import CodeParser
from doc_generator import DocGenerator

# Парсинг проекта
parser = CodeParser()
structure = parser.parse_project("/path/to/project")

# Генерация документации
generator = DocGenerator()
readme = generator.generate_readme(structure, "Project Name")

# Сохранение
with open("README.md", "w") as f:
    f.write(readme)
```

## Архитектура

Система состоит из трёх основных компонентов:

1. **CodeParser** - анализирует Python код через AST
2. **DocGenerator** - генерирует документацию через Claude API  
3. **FileWatcher** - отслеживает изменения файлов

**Пайплайн работы:**
```
Python проект → CodeParser → структура → DocGenerator → Claude API → README.md
```

**Автоматический режим:**
```
Изменение .py файла → FileWatcher → CodeParser → DocGenerator → обновлённый README
```

## API Reference

### CodeParser

Основной класс для парсинга Python кода.

```python
class CodeParser:
    def parse_file(self, file_path: str) -> dict:
        """Парсит один Python файл"""
        
    def parse_project(self, project_path: str) -> dict:
        """Парсит весь проект"""
        
    def get_summary(self) -> dict:
        """Краткая сводка о проекте"""
```

### DocGenerator

Генератор документации через Claude API.

```python
class DocGenerator:
    def generate_readme(self, project_structure: dict, project_name: str) -> str:
        """Генерирует README.md на основе структуры проекта
        
        Args:
            project_structure: словарь со структурой проекта
            project_name: название проекта
            
        Returns:
            текст README.md в формате Markdown
        """
        
    def generate_quick_summary(self, structure: dict) -> str:
        """Быстрая сводка о проекте (для превью)"""
```

### AutoDocsGenerator

Главный класс для генерации документации.

```python
class AutoDocsGenerator:
    def generate_for_project(self, project_path: str, output_path: str = None):
        """Генерирует документацию для одного проекта
        
        Args:
            project_path: путь к проекту
            output_path: куда сохранить README (по умолчанию в корень проекта)
        """
        
    def generate_batch(self, directory: str):
        """Генерирует документацию для всех проектов в директории
        
        Args:
            directory: папка с проектами
        """
```

### FileWatcher

Класс для автоматического отслеживания изменений.

```python
class FileWatcher:
    def __init__(self, watch_directory: str, interval: int = 5):
        """Args:
            watch_directory: папка для отслеживания
            interval: интервал проверки в секундах
        """
        
    def check_changes(self) -> list:
        """Проверяет какие файлы изменились"""
        
    def regenerate_docs(self, project_path: str):
        """Регенерирует документацию для проекта"""
        
    def start(self):
        """Запускает отслеживание"""
```

### Вспомогательные функции

```python
def demo_parse():
    """Демонстрация парсинга проекта"""
    
def main():
    """Главная функция с CLI интерфейсом"""
```