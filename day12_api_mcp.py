import os
import requests
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class WeatherTool:
    """Инструмент для получения погоды"""
    
    def __init__(self):
        self.name = "get_weather"
        self.description = "Получает текущую погоду для любого города"
    
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
                "admin1": result.get("admin1", ""),
            }
        except Exception:
            return None
    
    def get_weather(self, city: str) -> dict:
        """Получает погоду из Open-Meteo API"""
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
            
            full_name = location["name"]
            if location["admin1"]:
                full_name += f", {location['admin1']}"
            full_name += f", {location['country']}"
            
            return {
                "city": full_name,
                "temperature": weather["temperature"],
                "windspeed": weather["windspeed"],
                "description": weather_codes.get(weather["weathercode"], "неизвестно"),
                "coordinates": f"{location['latitude']}, {location['longitude']}",
                "time": datetime.now().strftime('%H:%M:%S')
            }
        except Exception as e:
            return {"error": str(e)}
    
    def call(self, city: str) -> str:
        """Вызов инструмента"""
        data = self.get_weather(city)
        
        if "error" in data:
            return f"Ошибка: {data['error']}"
        
        return f"""Текущая погода в {data['city']}:
- Температура: {data['temperature']}°C
- Ветер: {data['windspeed']} км/ч
- Условия: {data['description']}
- Координаты: {data['coordinates']}
- Время: {data['time']}
(Данные из Open-Meteo API)"""


class MCPServer:
    """MCP-сервер - регистрирует и предоставляет инструменты"""
    
    def __init__(self):
        self.tools = {
            "get_weather": WeatherTool()
        }
    
    def get_tool_definition(self):
        """Определение инструмента для Claude API"""
        return {
            "name": "get_weather",
            "description": "Получает текущую погоду для указанного города. Используйте когда пользователь спрашивает о погоде.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города (например: Warsaw, Moscow, Варшава)"
                    }
                },
                "required": ["city"]
            }
        }
    
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Вызов инструмента"""
        if tool_name in self.tools:
            city = arguments.get("city", "")
            return self.tools[tool_name].call(city)
        return f"Ошибка: инструмент {tool_name} не найден"


class ClaudeAgent:
    """Агент на Claude API - Claude САМ решает когда вызывать инструменты"""
    
    def __init__(self, api_key: str, mcp_server: MCPServer):
        self.client = Anthropic(api_key=api_key)
        self.mcp_server = mcp_server
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """Отправляет сообщение Claude"""
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[self.mcp_server.get_tool_definition()],
            messages=self.conversation_history
        )
        
        while response.stop_reason == "tool_use":
            tool_use = None
            assistant_message = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_use = block
                    print(f"\n[Claude] Вызов инструмента: {block.name}")
                    print(f"[Claude] Аргументы: {block.input}")
                assistant_message.append(block)
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            if tool_use:
                # Вызываем инструмент через MCP
                print(f"[MCP] Обработка запроса...")
                tool_result = self.mcp_server.call_tool(
                    tool_name=tool_use.name,
                    arguments=tool_use.input
                )
                print(f"[MCP] Результат получен")
                
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
                    max_tokens=1024,
                    tools=[self.mcp_server.get_tool_definition()],
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


def main():
    """Главная функция"""
        
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Ошибка: API ключ не найден!")
        print("\nСоздайте файл .env:")
        print("   ANTHROPIC_API_KEY=your-key-here\n")
        api_key = input("Или введите ключ: ").strip()
        
        if not api_key:
            print("API ключ обязателен!")
            return
        
    # Создаём компоненты
    mcp_server = MCPServer()
    print(f"Зарегистрированные инструменты: {list(mcp_server.tools.keys())}")
    
    agent = ClaudeAgent(api_key=api_key, mcp_server=mcp_server)
    
    print("="*70)
    print("ИНТЕРАКТИВНЫЙ ЧАТ")
    print("="*70)
    print("\nПримеры вопросов:")
    print("   - Какая погода в Варшаве?")
    print("   - Как в Токио сейчас?")
    print("   - Расскажи про Python\n")
    print("Введите 'exit' для выхода\n")
    print("="*70 + "\n")
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                print("\nДо свидания!")
                break
            
            response = agent.chat(user_input)
            print(f"\nClaude: {response}\n")
            print("-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":

    main()
