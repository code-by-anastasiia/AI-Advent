"""
День 14: Агент с композицией MCP
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MCPCompositionAgent:
    """Агент с композицией двух MCP-серверов"""
    
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # print("✅ Claude API подключен")
        
        # Определяем директорию скрипта
        self.script_dir = Path(__file__).parent.absolute()
        # print(f"📁 Директория скрипта: {self.script_dir}")
    
    async def process_query(self, query: str):
        """Обработка запроса через композицию MCP"""
        
        print("\n" + "="*70)
        print(f"ЗАПРОС: {query}")
        print("="*70)
        
        # Получаем путь к Python
        python_path = sys.executable
        # print(f"\nИспользуем Python: {python_path}")
        
        # Пути к MCP-серверам
        search_server_path = self.script_dir / "search_mcp_fixed.py"
        file_server_path = self.script_dir / "file_mcp_fixed.py"
        
        # Проверяем что файлы существуют
        if not search_server_path.exists():
            raise FileNotFoundError(f"❌ Не найден: {search_server_path}")
        if not file_server_path.exists():
            raise FileNotFoundError(f"❌ Не найден: {file_server_path}")
        
        print(f"✅ Найден: {search_server_path.name}")
        print(f"✅ Найден: {file_server_path.name}")
        
        # Параметры для MCP-серверов
        search_server = StdioServerParameters(
            command=python_path,
            args=[str(search_server_path)],
            env=None
        )
        
        file_server = StdioServerParameters(
            command=python_path,
            args=[str(file_server_path)],
            env=None
        )
        
        # ШАГ 1: Поиск через Web Search MCP
        print("\n[ШАГ 1] Подключение к Web Search MCP...")
        print(f"Запуск: {python_path} {search_server_path.name}")
        
        try:
            async with stdio_client(search_server) as (read, write):
                print("✅ Процесс запущен")
                
                async with ClientSession(read, write) as session:
                    print("Инициализация сессии...")
                    await session.initialize()
                    
                    print("✅ Подключено к Web Search MCP")
                    print("Выполняю поиск...")
                    
                    # Вызываем инструмент
                    result = await session.call_tool(
                        "web_search",
                        arguments={"query": query, "num_results": 5}
                    )
                    
                    # Парсим результат
                    search_data = json.loads(result.content[0].text)
                    
                    print(f"✅ Найдено результатов: {len(search_data.get('results', []))}")
                    
                    for i, res in enumerate(search_data.get("results", [])[:3], 1):
                        print(f"  {i}. {res.get('title', 'Без названия')}")
        
        except Exception as e:
            print(f"❌ Ошибка Web Search MCP: {e}")
            log_file = self.script_dir / "search_mcp_debug.log"
            print(f"\n💡 Проверьте лог файл: {log_file}")
            if log_file.exists():
                print("\nПоследние строки лога:")
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
            raise
        
        # ШАГ 2: Суммаризация через Claude
        print("\n[ШАГ 2] Суммаризация результатов через Claude...")
        
        formatted_results = self._format_search_results(search_data)
        
        summary_prompt = f"""
Вот результаты поиска по запросу "{query}":

{formatted_results}

Сделай краткое и информативное резюме на русском языке:
1. Основная информация по теме
2. Ключевые факты (3-5 пунктов)
3. Источники

Формат: структурированный текст, легко читаемый.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary = response.content[0].text
        
        print("✅ Summary создан")
        print("\nSUMMARY:")
        print("-" * 70)
        print(summary[:300] + "..." if len(summary) > 300 else summary)
        print("-" * 70)
        
        # ШАГ 3: Сохранение через File Saver MCP
        print("\n[ШАГ 3] Подключение к File Saver MCP...")
        print(f"Запуск: {python_path} {file_server_path.name}")
        
        final_doc = f"""
ИССЛЕДОВАНИЕ: {query}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{summary}

---
Создано автоматически через композицию двух MCP-серверов:
- Web Search MCP (поиск в интернете)
- File Saver MCP (сохранение в файл)
"""
        
        try:
            async with stdio_client(file_server) as (read, write):
                print("✅ Процесс запущен")
                
                async with ClientSession(read, write) as session:
                    print("Инициализация сессии...")
                    await session.initialize()
                    
                    print("✅ Подключено к File Saver MCP")
                    print("Сохраняю результат...")
                    
                    filename = self._generate_filename(query)
                    
                    # Вызываем инструмент
                    result = await session.call_tool(
                        "save_to_file",
                        arguments={"content": final_doc, "filename": filename}
                    )
                    
                    # Парсим результат
                    save_data = json.loads(result.content[0].text)
                    
                    if save_data.get("success"):
                        print(f"✅ Сохранено: {save_data['filename']}")
                        print(f"📁 Путь: {save_data['filepath']}")
                    else:
                        print(f"❌ Ошибка: {save_data.get('error')}")
        
        except Exception as e:
            print(f"❌ Ошибка File Saver MCP: {e}")
            log_file = self.script_dir / "file_mcp_debug.log"
            print(f"\n💡 Проверьте лог файл: {log_file}")
            if log_file.exists():
                print("\nПоследние строки лога:")
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:
                        print(f"  {line.rstrip()}")
            raise
        
        print("\n" + "="*70)
        print("КОМПОЗИЦИЯ ЗАВЕРШЕНА!")
        print("="*70)
        print("\n💡 Что произошло:")
        print("  1. MCP #1 (Web Search) запущен как отдельный процесс")
        print("  2. Агент подключился через stdio протокол")
        print("  3. Claude проанализировал результаты")
        print("  4. MCP #2 (File Saver) запущен как отдельный процесс")
        print("  5. Результат сохранён в файл")
        print("="*70 + "\n")
    
    def _format_search_results(self, results: dict) -> str:
        """Форматирует результаты поиска"""
        formatted = []
        
        for i, result in enumerate(results.get("results", []), 1):
            formatted.append(f"""
Результат {i}:
Заголовок: {result.get("title", "")}
Описание: {result.get("snippet", "")}
URL: {result.get("url", "")}
""")
        
        return "\n".join(formatted)
    
    def _generate_filename(self, query: str) -> str:
        """Генерирует имя файла"""
        words = query.lower().split()[:3]
        filename = "_".join(words)
        filename = "".join(c for c in filename if c.isalnum() or c == "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{filename}_{timestamp}.txt"


async def main():
    """Главная функция"""
    
    print("\n" + "="*70)
    print("ДЕНЬ 14: КОМПОЗИЦИЯ MCP")
    print("="*70)
    print("\nАрхитектура:")
    print("  Процесс 1: Agent")
    print("  Процесс 2: Search MCP Server")
    print("  Процесс 3: File MCP Server")
    print("="*70 + "\n")
    
    # Создаём агента
    agent = MCPCompositionAgent()
    
    # Примеры запросов
    queries = [
        "Новости про искусственный интеллект 2025",
        "Python MCP protocol",
        "Claude AI capabilities"
    ]
    
    print("\nВыберите запрос:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    print("0. Свой запрос")
    
    try:
        choice = input("\nВаш выбор (1-3 или 0): ").strip()
        
        if choice == "0":
            query = input("Введите запрос: ").strip()
        elif choice in ["1", "2", "3"]:
            query = queries[int(choice) - 1]
        else:
            query = queries[0]
        
        if query:
            await agent.process_query(query)
        else:
            print("Пустой запрос!")
            
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n💡 Убедитесь что все файлы в одной папке:")
        print("  - search_mcp_fixed.py")
        print("  - file_mcp_fixed.py")
        print("  - day14_auto_agent.py")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
