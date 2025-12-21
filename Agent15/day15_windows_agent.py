
import os
import sys
import json
import asyncio
from pathlib import Path
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


class WindowsProgramsAgent:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.script_dir = Path(__file__).parent.absolute()
        print("✅ Windows Programs Agent запущен\n")
    
    async def execute_command(self, user_input: str):
        print(f"Команда: {user_input}\n")
        
        python_path = sys.executable
        programs_mcp_path = self.script_dir / "windows_programs_mcp.py"
        
        if not programs_mcp_path.exists():
            raise FileNotFoundError(f"❌ Не найден: {programs_mcp_path}")
        
        programs_server = StdioServerParameters(
            command=python_path,
            args=[str(programs_mcp_path)]
        )
        
        print("[1/2] Подключение к Windows Programs MCP...")
        
        async with stdio_client(programs_server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("      ✅ Подключено\n")
                
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                tools_description = "\n".join([
                    f"- {tool.name}: {tool.description}"
                    for tool in tools
                ])
                
                system_prompt = f"""Ты помощник для управления программами Windows.

Инструменты:
{tools_description}

Когда пользователь просит:
1. Определи какой инструмент нужен
2. Вызови его с правильными параметрами
3. Объясни результат на русском

Примеры:
- "Открой блокнот" → open_program с program_name="notepad"
- "Открой калькулятор" → open_program с program_name="calculator"
- "Закрой блокнот" → close_program с program_name="notepad"
- "Что запущено?" → list_programs
"""
                
                print("[2/2] Claude анализирует...")
                
                messages = [{"role": "user", "content": user_input}]
                
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    tools=[{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools],
                    messages=messages
                )
                
                print("      ✅ Готово\n")
                
                tool_calls = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append(block)
                
                if tool_calls:
                    print("="*60)
                    print("ВЫПОЛНЕНИЕ")
                    print("="*60 + "\n")
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.input
                        
                        print(f"Инструмент: {tool_name}")
                        print(f"Параметры: {json.dumps(tool_args, ensure_ascii=False)}\n")
                        
                        result = await session.call_tool(tool_name, tool_args)
                        result_data = json.loads(result.content[0].text)
                        
                        print("Результат:")
                        print(json.dumps(result_data, ensure_ascii=False, indent=2))
                        print()
                        
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": result.content[0].text
                            }]
                        })
                    
                    print("="*60)
                    print("ОТВЕТ CLAUDE")
                    print("="*60 + "\n")
                    
                    final_response = self.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=messages
                    )
                    
                    for block in final_response.content:
                        if block.type == "text":
                            print(block.text)
                    
                    print("\n" + "="*60 + "\n")


async def main():
    print("\n" + "="*60)
    print("Windows Programs Agent")
    print("="*60 + "\n")
    
    agent = WindowsProgramsAgent()
    
    examples = [
        "Открой блокнот",
        "Открой калькулятор",
        "Покажи запущенные программы",
        "Закрой блокнот",
    ]
    
    print("Примеры команд:")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")
    print("\n0. Свой запрос\n")
    
    try:
        choice = input("Ваш выбор (1-4 или 0): ").strip()
        
        if choice == "0":
            command = input("Введите команду: ").strip()
        elif choice in ["1", "2", "3", "4"]:
            command = examples[int(choice) - 1]
        else:
            command = examples[0]
        
        if command:
            await agent.execute_command(command)
            
    except KeyboardInterrupt:
        print("\n\nПрервано")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
