"""
Демо версия - показывает работу парсера без Claude API
Можно запустить без API ключа
"""

import json
from pathlib import Path
from code_parser import CodeParser

def demo_parse():
    """Демонстрация парсинга проекта"""
    
    print("\n" + "="*80)
    print("📚 ДЕМО: Анализ структуры проекта")
    print("="*80)
    
    # Парсим study_progress_analyzer как пример
    project_path = r"C:\Users\atser\Desktop\Advent\day 24"
    
    if not Path(project_path).exists():
        print(f"❌ Проект не найден: {project_path}")
        return
    
    # Парсинг
    parser = CodeParser(project_path)
    structure = parser.parse_project()
    
    # Выводим результаты
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*80)
    
    print(f"\n🎯 {parser.get_summary()}\n")
    
    print("📚 Найденные классы:")
    for cls in structure["classes"]:
        print(f"\n  Класс: {cls['name']} ({cls['file']})")
        if cls['docstring']:
            print(f"  └─ {cls['docstring'][:80]}...")
        if cls['methods']:
            print(f"  └─ Методы: {', '.join([m['name'] for m in cls['methods'][:5]])}")
    
    print("\n🔧 Найденные функции:")
    for func in structure["functions"][:10]:
        print(f"  • {func['name']}() в {func['file']}")
        if func['docstring']:
            print(f"    └─ {func['docstring'][:60]}...")
    
    print(f"\n📦 Зависимости ({len(structure['imports'])}):")
    imports = sorted(structure['imports'])[:15]
    print(f"  {', '.join(imports)}")
    if len(structure['imports']) > 15:
        print(f"  ... и ещё {len(structure['imports']) - 15}")
    
    print("\n📄 Структура файлов:")
    for file_info in structure["files"]:
        print(f"\n  {file_info['path']}")
        if file_info['docstring']:
            print(f"  └─ {file_info['docstring'][:70]}")
        if file_info['classes']:
            print(f"  └─ Классы: {', '.join([c['name'] for c in file_info['classes']])}")
        if file_info['functions']:
            funcs = ', '.join([f['name'] for f in file_info['functions'][:3]])
            print(f"  └─ Функции: {funcs}")
    
    # Сохраняем структуру в JSON для просмотра
    output_file = "examples/parsed_structure.json"
    Path("examples").mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print(f"✅ Полная структура сохранена в: {output_file}")
    print("="*80)
    
    print("\n💡 Эта структура будет отправлена в Claude для генерации README")
    print("   Запусти main.py с API ключом чтобы увидеть полную генерацию")


if __name__ == "__main__":
    demo_parse()
