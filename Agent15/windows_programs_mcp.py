"""
MCP Server: Windows Programs Manager
Управление программами Windows
"""

import json
import subprocess
import asyncio
import psutil
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


server = Server("windows-programs-mcp")


def run_program(program_name: str) -> dict:
    """Запустить программу"""
    try:
        # Карта популярных программ
        programs = {
            "notepad": "notepad.exe",
            "блокнот": "notepad.exe",
            "calculator": "calc.exe",
            "калькулятор": "calc.exe",
            "paint": "mspaint.exe",
            "paint": "mspaint.exe",
            "explorer": "explorer.exe",
            "проводник": "explorer.exe",
        }
        
        # Получаем имя исполняемого файла
        exe_name = programs.get(program_name.lower(), program_name + ".exe")
        
        # Запускаем программу
        subprocess.Popen(exe_name, shell=True)
        
        return {
            "success": True,
            "message": f"Программа запущена: {program_name}",
            "executable": exe_name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def close_program(program_name: str) -> dict:
    """Закрыть программу"""
    try:
        programs = {
            "notepad": "notepad.exe",
            "блокнот": "notepad.exe",
            "calculator": "calc.exe",
            "калькулятор": "Calculator.exe",
            "paint": "mspaint.exe",
            "паинт": "mspaint.exe",
        }
        
        exe_name = programs.get(program_name.lower(), program_name + ".exe")
        
        # Ищем процесс
        closed_count = 0
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'].lower() == exe_name.lower():
                    proc.terminate()
                    closed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if closed_count > 0:
            return {
                "success": True,
                "message": f"Закрыто процессов: {closed_count}",
                "program": program_name
            }
        else:
            return {
                "success": False,
                "message": f"Программа {program_name} не запущена"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def list_programs() -> dict:
    """Показать запущенные программы"""
    try:
        # Список популярных программ для отслеживания
        track_programs = [
            "notepad.exe",
            "calc.exe",
            "Calculator.exe",
            "mspaint.exe",
            "explorer.exe",
            "chrome.exe",
            "firefox.exe",
            "code.exe"
        ]
        
        running = []
        
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                proc_name = proc.info['name']
                if proc_name in track_programs:
                    running.append({
                        "name": proc_name,
                        "pid": proc.info['pid']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            "success": True,
            "programs": running,
            "count": len(running)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Список инструментов"""
    return [
        Tool(
            name="open_program",
            description="Открыть программу Windows (notepad, calculator, paint, explorer)",
            inputSchema={
                "type": "object",
                "properties": {
                    "program_name": {
                        "type": "string",
                        "description": "Название программы (notepad, calculator, paint)"
                    }
                },
                "required": ["program_name"]
            }
        ),
        Tool(
            name="close_program",
            description="Закрыть программу",
            inputSchema={
                "type": "object",
                "properties": {
                    "program_name": {
                        "type": "string",
                        "description": "Название программы для закрытия"
                    }
                },
                "required": ["program_name"]
            }
        ),
        Tool(
            name="list_programs",
            description="Показать запущенные программы",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Вызов инструмента"""
    
    if name == "open_program":
        program_name = arguments.get("program_name")
        result = run_program(program_name)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    
    elif name == "close_program":
        program_name = arguments.get("program_name")
        result = close_program(program_name)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    
    elif name == "list_programs":
        result = list_programs()
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    
    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
