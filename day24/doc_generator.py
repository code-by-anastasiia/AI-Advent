"""
Генератор документации через Claude API
Принимает структуру проекта и создаёт README.md
"""

import json
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from code_parser import CodeParser

load_dotenv()


class DocGenerator:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_readme(self, project_structure: dict, project_name: str) -> str:
        """
        Генерирует README.md на основе структуры проекта
        
        Args:
            project_structure: словарь со структурой проекта
            project_name: название проекта
        
        Returns:
            текст README.md в формате Markdown
        """
        
        # Готовим данные для Claude
        structure_json = json.dumps(project_structure, ensure_ascii=False, indent=2)
        
        # System prompt
        system_prompt = """Ты - технический писатель, который создаёт документацию для Python проектов.

Твоя задача: на основе структуры проекта создать профессиональный README.md.

Структура README:
1. # Название проекта
2. ## Описание (2-3 предложения о назначении)
3. ## Структура проекта (дерево файлов с кратким описанием каждого)
4. ## Установка (команды для установки зависимостей)
5. ## Использование (примеры запуска)
6. ## Архитектура (как компоненты связаны)
7. ## API Reference (основные классы и функции с описанием)

Требования:
- Пиши на русском языке
- Будь конкретным, без воды
- Используй примеры кода где уместно
- Если есть docstrings - используй их
- Формат Markdown"""

        # Запрос к Claude
        print("\n🤖 Генерация документации через Claude...")
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.3,  # Низкая для точности
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"""Создай README.md для проекта "{project_name}".

Структура проекта:
{structure_json}

Создай полную профессиональную документацию."""
            }]
        )
        
        readme_content = response.content[0].text
        
        return readme_content
    
    def generate_quick_summary(self, project_structure: dict) -> str:
        """Быстрая сводка о проекте (для превью)"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.3,
            system="Ты создаёшь краткое описание Python проектов в 2-3 предложения.",
            messages=[{
                "role": "user",
                "content": f"""Опиши кратко (2-3 предложения) что делает этот проект:

Классы: {', '.join([c['name'] for c in project_structure.get('classes', [])])}
Функции: {', '.join([f['name'] for f in project_structure.get('functions', [])[:5]])}
Импорты: {', '.join(list(project_structure.get('imports', []))[:10])}"""
            }]
        )
        
        return response.content[0].text


if __name__ == "__main__":
    # Тест генератора
    from code_parser import CodeParser
    
    parser = CodeParser("/home/claude/study_progress_analyzer")
    structure = parser.parse_project()
    
    generator = DocGenerator()
    
    # Быстрая сводка
    summary = generator.generate_quick_summary(structure)
    print("\n📝 Краткое описание:")
    print(summary)
    
    # Полный README
    readme = generator.generate_readme(structure, "Study Progress Analyzer")
    print("\n" + "="*60)
    print("📄 СГЕНЕРИРОВАННЫЙ README:")
    print("="*60)
    print(readme[:500] + "...")
