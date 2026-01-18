"""
Автоматический генератор документации для Python проектов
День 24: Реальная задача - автоматизация разработки
"""

import os
import sys
from pathlib import Path
from code_parser import CodeParser
from doc_generator import DocGenerator


class AutoDocsGenerator:
    def __init__(self):
        self.parser = None
        self.generator = DocGenerator()
    
    def generate_for_project(self, project_path: str, output_path: str = None):
        """
        Генерирует документацию для одного проекта
        
        Args:
            project_path: путь к проекту
            output_path: куда сохранить README (по умолчанию в корень проекта)
        """
        
        print("\n" + "="*80)
        print("📚 ГЕНЕРАТОР ДОКУМЕНТАЦИИ")
        print("="*80)
        
        # Проверяем существование проекта
        project_path = Path(project_path)
        if not project_path.exists():
            print(f"❌ Путь не существует: {project_path}")
            return False
        
        # 1. Парсим код проекта
        print(f"\n🔍 Анализ проекта: {project_path.name}")
        self.parser = CodeParser(project_path)
        structure = self.parser.parse_project()
        
        print(f"\n✅ {self.parser.get_summary()}")
        
        # 2. Генерируем README
        readme_content = self.generator.generate_readme(
            structure, 
            project_path.name
        )
        
        # 3. Сохраняем
        if output_path is None:
            output_path = project_path / "README.md"
        else:
            output_path = Path(output_path)
        
        # Делаем бэкап если README уже существует
        if output_path.exists():
            backup_path = output_path.parent / "README_backup.md"
            print(f"\n💾 Бэкап существующего README: {backup_path}")
            output_path.rename(backup_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"\n✅ Документация сохранена: {output_path}")
        
        # 4. Показываем превью
        print("\n" + "="*80)
        print("📄 ПРЕВЬЮ ДОКУМЕНТАЦИИ")
        print("="*80)
        lines = readme_content.split('\n')
        print('\n'.join(lines[:30]))  # Первые 30 строк
        if len(lines) > 30:
            print(f"\n... и ещё {len(lines) - 30} строк")
        print("="*80)
        
        return True
    
    def generate_batch(self, directory: str):
        """
        Генерирует документацию для всех проектов в директории
        
        Args:
            directory: папка с проектами
        """
        
        directory = Path(directory)
        
        # Находим все подпапки с Python файлами
        projects = []
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Проверяем есть ли .py файлы
                py_files = list(item.rglob("*.py"))
                if py_files:
                    projects.append(item)
        
        print(f"\n📂 Найдено проектов: {len(projects)}")
        
        for i, project in enumerate(projects, 1):
            print(f"\n[{i}/{len(projects)}] Обработка: {project.name}")
            self.generate_for_project(project)
            print("\n" + "-"*80)
        
        print(f"\n✅ Завершено! Обработано проектов: {len(projects)}")


def main():
    """Главная функция с CLI интерфейсом"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         АВТОМАТИЧЕСКИЙ ГЕНЕРАТОР ДОКУМЕНТАЦИИ                 ║
║              Автоматизация разработки                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    generator = AutoDocsGenerator()
    
    print("\nВыберите режим:")
    print("1. Сгенерировать для одного проекта")
    print("2. Сгенерировать для всех проектов в папке")
    print("3. Тест на примере study_progress_analyzer")
    
    choice = input("\nВаш выбор (1/2/3): ").strip()
    
    if choice == "1":
        project_path = input("\nПуть к проекту: ").strip()
        generator.generate_for_project(project_path)
    
    elif choice == "2":
        directory = input("\nПуть к папке с проектами: ").strip()
        generator.generate_batch(directory)
    
    elif choice == "3":
        # Тестовый запуск
        test_project = "/home/claude/study_progress_analyzer"
        if Path(test_project).exists():
            generator.generate_for_project(test_project)
        else:
            print(f"❌ Тестовый проект не найден: {test_project}")
    
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    main()
