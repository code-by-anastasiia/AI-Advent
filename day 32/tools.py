import os
import json
import requests
from datetime import datetime

class ToolsManager:
    """Расширенные инструменты для God Agent"""
    
    def __init__(self):
        self.tasks_file = "tasks.json"
        self.notes_file = "notes.json"
    
    # === ЗАДАЧИ ===
    
    def add_task(self, title, priority="medium", deadline=None):
        """Добавить задачу с дедлайном"""
        tasks = self._load_json(self.tasks_file, [])
        
        new_task = {
            "id": len(tasks) + 1,
            "title": title,
            "priority": priority,
            "status": "pending",
            "created": datetime.now().isoformat(),
            "deadline": deadline
        }
        tasks.append(new_task)
        
        self._save_json(self.tasks_file, tasks)
        return f"✅ Задача #{new_task['id']} добавлена: {title}"
    
    def complete_task(self, task_id):
        """Отметить задачу как выполненную"""
        tasks = self._load_json(self.tasks_file, [])
        
        for task in tasks:
            if task['id'] == task_id:
                task['status'] = 'completed'
                task['completed'] = datetime.now().isoformat()
                self._save_json(self.tasks_file, tasks)
                return f"✅ Задача #{task_id} выполнена: {task['title']}"
        
        return f"❌ Задача #{task_id} не найдена"
    
    def get_tasks(self, status="all"):
        """Получить задачи"""
        tasks = self._load_json(self.tasks_file, [])
        
        if status != "all":
            tasks = [t for t in tasks if t['status'] == status]
        
        if not tasks:
            return f"📋 Задач ({status}) нет"
        
        result = f"📋 Задачи ({status}):\n\n"
        for task in tasks:
            deadline = f" | ⏰ {task['deadline']}" if task.get('deadline') else ""
            result += f"#{task['id']} [{task['priority']}] {task['title']}{deadline}\n"
        
        return result
    
    # === ЗАМЕТКИ ===
    
    def add_note(self, title, content, tags=None):
        """Добавить заметку"""
        notes = self._load_json(self.notes_file, [])
        
        new_note = {
            "id": len(notes) + 1,
            "title": title,
            "content": content,
            "tags": tags or [],
            "created": datetime.now().isoformat()
        }
        notes.append(new_note)
        
        self._save_json(self.notes_file, notes)
        return f"📝 Заметка #{new_note['id']} создана: {title}"
    
    def search_notes(self, query):
        """Найти заметки по тексту"""
        notes = self._load_json(self.notes_file, [])
        
        found = [n for n in notes if 
                 query.lower() in n['title'].lower() or 
                 query.lower() in n['content'].lower()]
        
        if not found:
            return "🔍 Заметок не найдено"
        
        result = f"🔍 Найдено заметок: {len(found)}\n\n"
        for note in found:
            result += f"#{note['id']} {note['title']}\n{note['content'][:100]}...\n\n"
        
        return result
    
    # === ПОГОДА ===
    
    def get_weather(self, city="Neumünster"):
        """Получить погоду"""
        try:
            # Геокодинг
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": city, "count": 1, "language": "ru", "format": "json"}
            geo_response = requests.get(geo_url, params=geo_params, timeout=5)
            geo_data = geo_response.json()
            
            if not geo_data.get('results'):
                return f"❌ Город {city} не найден"
            
            lat = geo_data['results'][0]['latitude']
            lon = geo_data['results'][0]['longitude']
            
            # Погода
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "auto"
            }
            weather_response = requests.get(weather_url, params=weather_params, timeout=5)
            weather_data = weather_response.json()
            
            current = weather_data['current_weather']
            
            return f"🌤️ Погода в {city}:\n" \
                   f"Температура: {current['temperature']}°C\n" \
                   f"Ветер: {current['windspeed']} км/ч"
        
        except Exception as e:
            return f"❌ Ошибка получения погоды: {e}"
    
    # === ВСПОМОГАТЕЛЬНЫЕ ===
    
    def _load_json(self, filename, default):
        """Загрузить JSON файл"""
        if not os.path.exists(filename):
            return default
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_json(self, filename, data):
        """Сохранить JSON файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
