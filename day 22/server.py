import json
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent

with open("crm_data.json", "r", encoding="utf-8") as f:
    crm_data = json.load(f)

app = Server("fittrack-crm")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_user_info",
            description="Получить данные пользователя: подписка, цели, устройства",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        ),
        Tool(
            name="get_user_tickets",
            description="Получить все тикеты пользователя",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        ),
        Tool(
            name="check_subscription",
            description="Проверить тип подписки и доступные функции",
            inputSchema={
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_user_info":
        user = crm_data["users"].get(arguments["user_id"])
        return [TextContent(type="text", text=json.dumps(user, ensure_ascii=False) if user else "Пользователь не найден")]
    
    elif name == "get_user_tickets":
        tickets = [
            {**ticket, "ticket_id": tid} 
            for tid, ticket in crm_data["tickets"].items() 
            if ticket["user_id"] == arguments["user_id"]
        ]
        return [TextContent(type="text", text=json.dumps(tickets, ensure_ascii=False))]
    
    elif name == "check_subscription":
        user = crm_data["users"].get(arguments["user_id"])
        if not user:
            return [TextContent(type="text", text="Пользователь не найден")]
        
        sub_info = {
            "subscription": user["subscription"],
            "features": {
                "free": ["1 устройство", "базовая статистика", "счётчик шагов"],
                "premium": ["неограниченно устройств", "персональные планы", "расширенная аналитика", "экспорт данных"]
            }[user["subscription"]]
        }
        return [TextContent(type="text", text=json.dumps(sub_info, ensure_ascii=False))]

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())