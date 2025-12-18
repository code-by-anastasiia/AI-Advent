"""
MCP Server #2: File Saver (исправленная версия с логированием)
"""

import json
import os
import asyncio
import sys
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Логирование в файл
def log(message):
    with open("file_mcp_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
        f.flush()


log("=== File MCP Server Starting ===")

# Создаём сервер
server = Server("file-saver-mcp")
log("Server object created")

# Папка для сохранения
OUTPUT_DIR = "saved_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
log(f"Output directory: {OUTPUT_DIR}")


def save_file_func(content: str, filename: str = None) -> dict:
    """Сохранить в файл"""
    log(f"Saving file: {filename or 'auto'}")
    
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"result_{timestamp}.txt"
        
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        abs_path = os.path.abspath(filepath)
        
        log(f"File saved: {abs_path}")
        
        return {
            "success": True,
            "filename": filename,
            "filepath": abs_path,
            "size": len(content)
        }
        
    except Exception as e:
        log(f"Save error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Список инструментов"""
    log("list_tools called")
    return [
        Tool(
            name="save_to_file",
            description="Сохранить текст в файл",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Содержимое для сохранения"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Имя файла (опционально)"
                    }
                },
                "required": ["content"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Вызов инструмента"""
    log(f"call_tool: {name}")
    
    if name == "save_to_file":
        content = arguments.get("content", "")
        filename = arguments.get("filename", None)
        
        result = save_file_func(content, filename)
        
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    
    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Запуск сервера"""
    log("Starting stdio_server...")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            log("stdio_server started, running server...")
            
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
            log("Server finished")
    except Exception as e:
        log(f"Server error: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    log(f"Python version: {sys.version}")
    log(f"Starting main()...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Server interrupted by user")
    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        log(traceback.format_exc())
