"""
Инструмент для работы с историей погоды
"""

import json
import os
from datetime import datetime
from typing import List, Dict


class WeatherHistoryTool:
    """Инструмент для сохранения и чтения истории погоды"""
    
    def __init__(self, json_file: str = "weather_history.json"):
        self.name = "weather_history"
        self.description = "Работа с историей погоды"
        self.json_file = json_file
        
        # Создать файл если не существует
        if not os.path.exists(self.json_file):
            self._save_data({"city": "Warsaw", "history": []})
    
    def _load_data(self) -> dict:
        """Загрузить данные из JSON"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка чтения {self.json_file}: {e}")
            return {"city": "Warsaw", "history": []}
    
    def _save_data(self, data: dict):
        """Сохранить данные в JSON"""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка записи {self.json_file}: {e}")
    
    def add_weather_record(self, city: str, temperature: float, 
                          windspeed: float, description: str) -> str:
        """Добавить запись о погоде"""
        data = self._load_data()
        
        record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "temperature": temperature,
            "windspeed": windspeed,
            "description": description
        }
        
        data["city"] = city
        data["history"].append(record)
        
        # Ограничиваем историю последними 100 записями
        if len(data["history"]) > 100:
            data["history"] = data["history"][-100:]
        
        self._save_data(data)
        
        return f"Запись добавлена. Всего записей: {len(data['history'])}"
    
    def get_history(self, limit: int = None) -> str:
        """Получить историю погоды"""
        data = self._load_data()
        history = data.get("history", [])
        
        if not history:
            return "История погоды пуста"
        
        if limit:
            history = history[-limit:]
        
        result = f"История погоды для {data.get('city', 'Unknown')}:\n"
        result += f"Всего записей: {len(data.get('history', []))}\n\n"
        
        if limit:
            result += f"Последние {len(history)} записей:\n"
        
        for record in history:
            result += f"- {record['timestamp']}: {record['temperature']}°C, "
            result += f"ветер {record['windspeed']} км/ч, {record['description']}\n"
        
        return result.strip()
    
    def get_statistics(self) -> str:
        """Получить статистику по истории"""
        data = self._load_data()
        history = data.get("history", [])
        
        if not history:
            return "История погоды пуста"
        
        temps = [r['temperature'] for r in history]
        winds = [r['windspeed'] for r in history]
        
        stats = {
            "city": data.get("city", "Unknown"),
            "total_records": len(history),
            "period_start": history[0]['timestamp'],
            "period_end": history[-1]['timestamp'],
            "temperature": {
                "min": min(temps),
                "max": max(temps),
                "avg": sum(temps) / len(temps)
            },
            "windspeed": {
                "min": min(winds),
                "max": max(winds),
                "avg": sum(winds) / len(winds)
            }
        }
        
        return json.dumps(stats, ensure_ascii=False, indent=2)
    
    def call(self, action: str, **kwargs) -> str:
        """
        Универсальный вызов инструмента
        
        Действия:
        - add: добавить запись (city, temperature, windspeed, description)
        - history: получить историю (limit - необязательно)
        - stats: получить статистику
        """
        
        if action == "add":
            city = kwargs.get('city', 'Unknown')
            temperature = kwargs.get('temperature', 0)
            windspeed = kwargs.get('windspeed', 0)
            description = kwargs.get('description', 'неизвестно')
            return self.add_weather_record(city, temperature, windspeed, description)
        
        elif action == "history":
            limit = kwargs.get('limit')
            return self.get_history(limit)
        
        elif action == "stats":
            return self.get_statistics()
        
        else:
            return f"Неизвестное действие: {action}"


if __name__ == "__main__":
    # Тест
    tool = WeatherHistoryTool()
    
    print("Добавление записей...")
    print(tool.add_weather_record("Warsaw", -2, 12, "облачно"))
    print(tool.add_weather_record("Warsaw", -1, 10, "малооблачно"))
    print(tool.add_weather_record("Warsaw", 0, 8, "ясно"))
    
    print("\nИстория:")
    print(tool.get_history())
    
    print("\nСтатистика:")
    print(tool.get_statistics())
