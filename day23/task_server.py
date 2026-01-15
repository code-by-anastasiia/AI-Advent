#!/usr/bin/env python3
"""
MCP Server для управления задачами курса AI агентов.
Предоставляет tools для чтения, создания и обновления задач.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Путь к файлу задач
TASKS_FILE = Path(__file__).parent / "tasks.json"

def load_tasks():
    """Загрузить задачи из JSON файла"""
    try:
        with open(TASKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tasks": []}

def save_tasks(data):
    """Сохранить задачи в JSON файл"""
    with open(TASKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Создаем MCP сервер
app = Server("task-manager")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Список доступных инструментов"""
    return [
        Tool(
            name="get_tasks",
            description="Получить список задач с фильтрацией по статусу и приоритету",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Фильтр по статусу: completed, in_progress, todo",
                        "enum": ["completed", "in_progress", "todo"]
                    },
                    "priority": {
                        "type": "string",
                        "description": "Фильтр по приоритету: high, medium, low",
                        "enum": ["high", "medium", "low"]
                    },
                    "tag": {
                        "type": "string",
                        "description": "Фильтр по тегу (например: rag, mcp, bug)"
                    }
                }
            }
        ),
        Tool(
            name="get_task_by_id",
            description="Получить детали конкретной задачи по ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID задачи"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="create_task",
            description="Создать новую задачу",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Название задачи"
                    },
                    "description": {
                        "type": "string",
                        "description": "Описание задачи"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Приоритет: high, medium, low",
                        "enum": ["high", "medium", "low"]
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Теги задачи (например: ['rag', 'bug'])"
                    }
                },
                "required": ["title", "priority"]
            }
        ),
        Tool(
            name="update_task_status",
            description="Обновить статус задачи",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID задачи"
                    },
                    "status": {
                        "type": "string",
                        "description": "Новый статус",
                        "enum": ["completed", "in_progress", "todo"]
                    }
                },
                "required": ["task_id", "status"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Обработка вызовов инструментов"""
    
    # Логируем в stderr (не в stdout!)
    print(f"[TOOL CALL] {name} with {arguments}", file=sys.stderr)
    
    if name == "get_tasks":
        # Фильтрация задач
        data = load_tasks()
        tasks = data.get("tasks", [])
        
        # Применяем фильтры
        if "status" in arguments:
            tasks = [t for t in tasks if t.get("status") == arguments["status"]]
        
        if "priority" in arguments:
            tasks = [t for t in tasks if t.get("priority") == arguments["priority"]]
        
        if "tag" in arguments:
            tag = arguments["tag"]
            tasks = [t for t in tasks if tag in t.get("tags", [])]
        
        result = {
            "count": len(tasks),
            "tasks": tasks
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    
    elif name == "get_task_by_id":
        # Получить задачу по ID
        task_id = arguments["task_id"]
        data = load_tasks()
        tasks = data.get("tasks", [])
        
        task = next((t for t in tasks if t["id"] == task_id), None)
        
        if task:
            return [TextContent(
                type="text",
                text=json.dumps(task, ensure_ascii=False, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Задача {task_id} не найдена"})
            )]
    
    elif name == "create_task":
        # Создать новую задачу
        data = load_tasks()
        tasks = data.get("tasks", [])
        
        # Генерируем новый ID
        new_id = max([t["id"] for t in tasks], default=0) + 1
        
        new_task = {
            "id": new_id,
            "title": arguments["title"],
            "status": "todo",
            "priority": arguments["priority"],
            "description": arguments.get("description", ""),
            "tags": arguments.get("tags", []),
            "created_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        tasks.append(new_task)
        data["tasks"] = tasks
        save_tasks(data)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "message": f"Задача #{new_id} создана",
                "task": new_task
            }, ensure_ascii=False, indent=2)
        )]
    
    elif name == "update_task_status":
        # Обновить статус задачи
        task_id = arguments["task_id"]
        new_status = arguments["status"]
        
        data = load_tasks()
        tasks = data.get("tasks", [])
        
        task = next((t for t in tasks if t["id"] == task_id), None)
        
        if task:
            old_status = task["status"]
            task["status"] = new_status
            
            # Добавляем дату завершения
            if new_status == "completed":
                task["completed_date"] = datetime.now().strftime("%Y-%m-%d")
            elif new_status == "in_progress":
                task["started_date"] = datetime.now().strftime("%Y-%m-%d")
            
            save_tasks(data)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Задача #{task_id}: {old_status} → {new_status}",
                    "task": task
                }, ensure_ascii=False, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Задача {task_id} не найдена"})
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]

async def main():
    """Запуск MCP сервера"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())