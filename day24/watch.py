"""
Автоматический режим - отслеживание изменений в проектах
При изменении .py файлов автоматически обновляет README
"""

import time
import os
from pathlib import Path
from datetime import datetime
from code_parser import CodeParser
from doc_generator import DocGenerator


class FileWatcher:
    def __init__(self, watch_directory: str, interval: int = 30):
        """
        Args:
            watch_directory: папка для отслеживания
            interval: интервал проверки в секундах
        """
        self.watch_dir = Path(watch_directory)
        self.interval = interval
        self.generator = DocGenerator()
        self.file_timestamps = {}
        
        # Инициализируем timestamps
        self._scan_files()
    
    def _scan_files(self):
        """Сканирует все Python файлы и запоминает время изменения"""
        for py_file in self.watch_dir.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                self.file_timestamps[py_file] = py_file.stat().st_mtime
    
    def check_changes(self) -> list:
        """Проверяет какие файлы изменились"""
        changed_projects = set()
        
        for py_file in self.watch_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            current_mtime = py_file.stat().st_mtime
            
            # Новый файл или изменённый
            if py_file not in self.file_timestamps or \
               current_mtime > self.file_timestamps[py_file]:
                
                # Находим корень проекта (папка где есть .py файлы)
                project_root = py_file.parent
                while project_root != self.watch_dir:
                    if (project_root / "README.md").exists() or \
                       len(list(project_root.glob("*.py"))) > 1:
                        break
                    project_root = project_root.parent
                
                changed_projects.add(project_root)
                self.file_timestamps[py_file] = current_mtime
        
        return list(changed_projects)
    
    def regenerate_docs(self, project_path: Path):
        """Регенерирует документацию для проекта"""
        
        print(f"\n🔄 Обнаружены изменения в: {project_path.name}")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # Парсим проект
            parser = CodeParser(project_path)
            structure = parser.parse_project()
            
            # Генерируем README
            readme = self.generator.generate_readme(structure, project_path.name)
            
            # Сохраняем
            readme_path = project_path / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme)
            
            print(f"✅ README обновлён: {readme_path}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    def start(self):
        """Запускает отслеживание"""
        
        print("\n" + "="*80)
        print("👀 АВТОМАТИЧЕСКИЙ РЕЖИМ")
        print("="*80)
        print(f"📂 Отслеживаемая папка: {self.watch_dir}")
        print(f"⏱️  Интервал проверки: {self.interval} сек")
        print("\n💡 Измени любой .py файл в проектах - README обновится автоматически")
        print("   Нажми Ctrl+C для остановки\n")
        
        try:
            while True:
                changed = self.check_changes()
                
                if changed:
                    for project in changed:
                        self.regenerate_docs(project)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Остановлено пользователем")


def main():
    """CLI для автоматического режима"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ ДОКУМЕНТАЦИИ                ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    watch_dir = input("Путь к папке с проектами: ").strip()
    
    if not Path(watch_dir).exists():
        print(f"❌ Папка не существует: {watch_dir}")
        return
    
    interval = input("Интервал проверки в секундах [30]: ").strip()
    interval = int(interval) if interval else 30
    
    watcher = FileWatcher(watch_dir, interval)
    watcher.start()


if __name__ == "__main__":
    main()
