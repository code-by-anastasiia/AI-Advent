"""
Парсер Python кода
Извлекает структуру проекта: функции, классы, импорты
"""

import ast
import os
from typing import Dict, List
from pathlib import Path


class CodeParser:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.structure = {
            "files": [],
            "imports": set(),
            "classes": [],
            "functions": [],
            "constants": []
        }
    
    def parse_file(self, filepath: Path) -> Dict:
        """Парсит один Python файл"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_info = {
                "path": str(filepath.relative_to(self.project_path)),
                "imports": [],
                "classes": [],
                "functions": [],
                "docstring": ast.get_docstring(tree) or ""
            }
            
            for node in ast.walk(tree):
                # Импорты
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info["imports"].append(alias.name)
                        self.structure["imports"].add(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_info["imports"].append(node.module)
                        self.structure["imports"].add(node.module)
                
                # Классы
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": []
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info["methods"].append({
                                "name": item.name,
                                "docstring": ast.get_docstring(item) or ""
                            })
                    
                    file_info["classes"].append(class_info)
                    self.structure["classes"].append({
                        "file": str(filepath.relative_to(self.project_path)),
                        **class_info
                    })
                
                # Функции (не методы)
                elif isinstance(node, ast.FunctionDef):
                    # Проверяем что это не метод класса
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            if node in parent.body:
                                is_method = True
                                break
                    
                    if not is_method:
                        func_info = {
                            "name": node.name,
                            "docstring": ast.get_docstring(node) or "",
                            "args": [arg.arg for arg in node.args.args]
                        }
                        file_info["functions"].append(func_info)
                        self.structure["functions"].append({
                            "file": str(filepath.relative_to(self.project_path)),
                            **func_info
                        })
            
            return file_info
            
        except Exception as e:
            print(f"❌ Ошибка парсинга {filepath}: {e}")
            return None
    
    def parse_project(self) -> Dict:
        """Парсит весь проект"""
        print(f"\n📂 Анализ проекта: {self.project_path}")
        
        # Ищем все Python файлы
        python_files = list(self.project_path.rglob("*.py"))
        
        # Исключаем __pycache__ и виртуальные окружения
        python_files = [
            f for f in python_files 
            if "__pycache__" not in str(f) and "venv" not in str(f)
        ]
        
        print(f"📄 Найдено файлов: {len(python_files)}")
        
        for filepath in python_files:
            file_info = self.parse_file(filepath)
            if file_info:
                self.structure["files"].append(file_info)
        
        # Конвертируем set в list для JSON
        self.structure["imports"] = list(self.structure["imports"])
        
        # Ищем requirements.txt
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                self.structure["requirements"] = f.read().strip()
        
        # Ищем README.md
        readme_file = self.project_path / "README.md"
        if readme_file.exists():
            with open(readme_file, 'r', encoding='utf-8') as f:
                self.structure["existing_readme"] = f.read()
        
        return self.structure
    
    def get_summary(self) -> str:
        """Краткая сводка о проекте"""
        return f"""
Проект: {self.project_path.name}
Файлов: {len(self.structure['files'])}
Классов: {len(self.structure['classes'])}
Функций: {len(self.structure['functions'])}
Зависимостей: {len(self.structure['imports'])}
        """.strip()


if __name__ == "__main__":
    # Тест на примере
    parser = CodeParser("/home/claude/study_progress_analyzer")
    structure = parser.parse_project()
    
    print("\n" + "="*60)
    print(parser.get_summary())
    print("="*60)
    
    print("\n📚 Классы:")
    for cls in structure["classes"][:3]:
        print(f"  - {cls['name']} ({cls['file']})")
    
    print("\n🔧 Функции:")
    for func in structure["functions"][:5]:
        print(f"  - {func['name']}() в {func['file']}")
