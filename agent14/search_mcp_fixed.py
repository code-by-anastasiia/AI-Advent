"""
MCP Server #1: Web Search (исправленная версия с логированием)
"""

import json
import requests
import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Логирование в файл (чтобы видеть ошибки)
def log(message):
    with open("search_mcp_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
        f.flush()


log("=== Search MCP Server Starting ===")

# Создаём сервер
server = Server("web-search-mcp")
log("Server object created")


def search_duckduckgo(query: str, num_results: int = 5) -> dict:
    """Поиск через DuckDuckGo API"""
    log(f"Searching for: {query}")
    
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = {"query": query, "results": []}
        
        if data.get("Abstract"):
            results["results"].append({
                "title": data.get("Heading", ""),
                "snippet": data.get("Abstract", ""),
                "url": data.get("AbstractURL", "")
            })
        
        for topic in data.get("RelatedTopics", [])[:num_results-1]:
            if isinstance(topic, dict) and "Text" in topic:
                results["results"].append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", "")
                })
        
        if not results["results"]:
            results["results"].append({
                "title": "Информация не найдена",
                "snippet": f"По запросу '{query}' информации не найдено.",
                "url": ""
            })
        
        log(f"Search completed: {len(results['results'])} results")
        return results
        
    except Exception as e:
        log(f"Search error: {e}")
        return {
            "query": query,
            "error": str(e),
            "results": [{
                "title": "Ошибка поиска",
                "snippet": f"Ошибка: {e}",
                "url": ""
            }]
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Список инструментов"""
    log("list_tools called")
    return [
        Tool(
            name="web_search",
            description="Поиск информации в интернете через DuckDuckGo",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Количество результатов (по умолчанию 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Вызов инструмента"""
    log(f"call_tool: {name} with args: {arguments}")
    
    if name == "web_search":
        query = arguments.get("query", "")
        num_results = arguments.get("num_results", 5)
        
        results = search_duckduckgo(query, num_results)
        
        return [TextContent(
            type="text",
            text=json.dumps(results, ensure_ascii=False, indent=2)
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
