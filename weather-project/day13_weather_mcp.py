"""
День 13: MCP-сервер с погодой и историей
Версия как у Владимира: собирает погоду → сохраняет → делает summary
"""

import os
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from weather_history_tool import WeatherHistoryTool

load_dotenv()


class WeatherTool:
    """Инструмент для получения текущей погоды"""
    
    def __init__(self):
        self.name = "get_weather"
        self.description = "Получает текущую погоду"
    
    def geocode_city(self, city_name: str) -> dict:
        """Находит координаты города"""
        try:
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {"name": city_name, "count": 1, "language": "ru", "format": "json"}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if "results" not in data or len(data["results"]) == 0:
                return None
            
            result = data["results"][0]
            return {
                "name": result["name"],
                "country": result.get("country", "Unknown"),
                "latitude": result["latitude"],
                "longitude": result["longitude"],
            }
        except Exception:
            return None
    
    def get_weather(self, city: str) -> dict:
        """Получает погоду из API"""
        try:
            location = self.geocode_city(city)
            if location is None:
                return {"error": f"Город '{city}' не найден"}
            
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current_weather": True,
                "timezone": "auto"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather = data["current_weather"]
            
            weather_codes = {
                0: "ясно", 1: "малооблачно", 2: "облачно",
                3: "пасмурно", 61: "дождь", 73: "снег", 95: "гроза"
            }
            
            return {
                "city": location["name"],
                "temperature": weather["temperature"],
                "windspeed": weather["windspeed"],
                "description": weather_codes.get(weather["weathercode"], "неизвестно"),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def call(self, city: str) -> str:
        """Вызов инструмента"""
        data = self.get_weather(city)
        
        if "error" in data:
            return f"Ошибка: {data['error']}"
        
        # Возвращаем структурированные данные для дальнейшей обработки
        import json
        return json.dumps(data, ensure_ascii=False)


class MCPServer:
    """MCP-сервер с погодой и историей"""
    
    def __init__(self):
        self.tools = {
            "get_weather": WeatherTool(),
            "weather_history": WeatherHistoryTool()
        }
    
    def get_tool_definitions(self):
        """Определения всех инструментов для Claude API"""
        return [
            {
                "name": "get_weather",
                "description": "Получает текущую погоду для города",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Название города (например: Warsaw, Moscow)"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "weather_history",
                "description": "Управление историей погоды. Действия: add (добавить), history (получить историю), stats (статистика)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Действие: 'add', 'history', 'stats'",
                            "enum": ["add", "history", "stats"]
                        },
                        "city": {
                            "type": "string",
                            "description": "Название города (для action=add)"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Температура в °C (для action=add)"
                        },
                        "windspeed": {
                            "type": "number",
                            "description": "Скорость ветра в км/ч (для action=add)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Описание погоды (для action=add)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Количество записей (для action=history)"
                        }
                    },
                    "required": ["action"]
                }
            }
        ]
    
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Вызов инструмента"""
        if tool_name == "get_weather":
            city = arguments.get("city", "")
            return self.tools["get_weather"].call(city)
        
        elif tool_name == "weather_history":
            return self.tools["weather_history"].call(**arguments)
        
        else:
            return f"Ошибка: инструмент {tool_name} не найден"


class ClaudeAgent:
    """Агент на Claude API"""
    
    def __init__(self, api_key: str, mcp_server: MCPServer):
        self.client = Anthropic(api_key=api_key)
        self.mcp_server = mcp_server
        self.conversation_history = []
    
    def chat(self, user_message: str, silent: bool = False) -> str:
        """Отправляет сообщение Claude"""
        
        # Добавляем сообщение пользователя
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Вызываем Claude API
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=self.mcp_server.get_tool_definitions(),
            messages=self.conversation_history
        )
        
        # Обрабатываем вызовы инструментов
        while response.stop_reason == "tool_use":
            tool_use = None
            assistant_message = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_use = block
                    if not silent:
                        print(f"[Claude] Вызов: {block.name}")
                assistant_message.append(block)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            if tool_use:
                # Вызываем инструмент через MCP
                tool_result = self.mcp_server.call_tool(
                    tool_name=tool_use.name,
                    arguments=tool_use.input
                )
                
                if not silent:
                    print(f"[MCP] Готово")
                
                # Отправляем результат Claude
                self.conversation_history.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result
                    }]
                })
                
                # Получаем финальный ответ
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    tools=self.mcp_server.get_tool_definitions(),
                    messages=self.conversation_history
                )
        
        # Извлекаем текстовый ответ
        final_response = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_response += block.text
        
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response
        })
        
        return final_response


def create_agent():
    """Создаёт агента с MCP-сервером"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Ошибка: API ключ не найден!")
        return None
    
    mcp_server = MCPServer()
    agent = ClaudeAgent(api_key=api_key, mcp_server=mcp_server)
    
    return agent
