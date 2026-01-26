import json
import os
import http.client
from dotenv import load_dotenv

# Загрузка API ключа
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ API ключ не найден в .env файле!")
    print("Создай файл .env и добавь: ANTHROPIC_API_KEY=твой_ключ")
    exit(1)

class PersonalAgent:
    def __init__(self, profile_path="user_profile.json"):
        self.api_key = api_key
        self.profile_path = profile_path
        self.profile = self.load_profile()
        self.conversation_history = []
    
    def load_profile(self):
        if not os.path.exists(self.profile_path):
            return self.create_default_profile()
        
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_default_profile(self):
        default = {
            "personal": {
                "name": "Miranda",
                "role": "студентка информатики",
                "location": "Германия",
                "education": "3 курс CS"
            },
            "tech_stack": {
                "languages": ["Python"],
                "frameworks": ["FastAPI", "Streamlit"],
                "tools": ["VS Code", "Ollama", "Git"],
                "ai_tools": ["Claude API", "gemma2", "gemma3"]
            },
            "learning_style": {
                "explanation_preference": "пошаговые примеры с кодом",
                "language": "Russian",
                "detail_level": "средний",
                "code_comments": "подробные"
            },
            "current_projects": {
                "main": "курс по AI-агентам день 30",
                "completed": [
                    "литературный критик на VPS",
                    "локальный аналитик данных",
                    "team assistant с RAG+MCP"
                ]
            },
            "interests": {
                "primary": ["multi-agent системы", "медицинские AI приложения"],
                "career_goal": "AI Engineer / LLM Application Developer"
            }
        }
        
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Создан профиль: {self.profile_path}\n")
        return default
    
    def build_system_prompt(self):
        p = self.profile
        return f"""Ты персональный AI-ассистент для {p['personal']['name']}.

Профиль пользователя:
- Роль: {p['personal']['role']}
- Образование: {p['personal']['education']}
- Локация: {p['personal']['location']}

Технический стек:
- Языки: {', '.join(p['tech_stack']['languages'])}
- Фреймворки: {', '.join(p['tech_stack']['frameworks'])}
- Инструменты: {', '.join(p['tech_stack']['tools'])}
- AI инструменты: {', '.join(p['tech_stack']['ai_tools'])}

Стиль обучения:
- Предпочтения: {p['learning_style']['explanation_preference']}
- Уровень детализации: {p['learning_style']['detail_level']}
- Комментарии в коде: {p['learning_style']['code_comments']}

Текущий проект: {p['current_projects']['main']}

Интересы: {', '.join(p['interests']['primary'])}
Карьерная цель: {p['interests']['career_goal']}

Правила общения:
1. Начинай с простого примера
2. Используй аналогии из реальной жизни
3. Показывай код с подробными комментариями
4. Объясняй практическое применение
5. Отвечай кратко по существу, не упуская важного
6. Не повторяй одно и то же разными словами
"""
    
    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2048,
            "system": self.build_system_prompt(),
            "messages": self.conversation_history
        }
        
        try:
            conn = http.client.HTTPSConnection("api.anthropic.com")
            
            headers = {
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            }
            
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            
            conn.request("POST", "/v1/messages", body, headers)
            response = conn.getresponse()
            data = response.read().decode('utf-8')
            
            result = json.loads(data)
            assistant_message = result["content"][0]["text"]
            
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            conn.close()
            return assistant_message
            
        except Exception as e:
            return f"Ошибка: {e}"
    
    def show_profile(self):
        print("\n" + "="*60)
        print("👤 ВАШ ПРОФИЛЬ")
        print("="*60)
        print(json.dumps(self.profile, ensure_ascii=False, indent=2))
        print("="*60 + "\n")
    
    def clear_history(self):
        self.conversation_history = []
        print("🗑️ История диалога очищена\n")


def main():
    print("="*60)
    print("🤖 ПЕРСОНАЛЬНЫЙ AI-АССИСТЕНТ")
    print("="*60 + "\n")
    
    agent = PersonalAgent()
    
    print(f"✅ Загружен профиль: {agent.profile['personal']['name']}")
    print(f"📚 Текущий проект: {agent.profile['current_projects']['main']}")
    
    print("\nКоманды:")
    print("  /profile - показать твой профиль")
    print("  /clear - очистить историю")
    print("  /exit - выход")
    print("="*60 + "\n")
    
    while True:
        user_input = input("Ты: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["/exit", "exit", "выход"]:
            print("👋 До встречи!")
            break
        
        if user_input == "/profile":
            agent.show_profile()
            continue
        
        if user_input == "/clear":
            agent.clear_history()
            continue
        
        print("\n🤖 Ассистент:")
        response = agent.chat(user_input)
        print(response + "\n")


if __name__ == "__main__":
    main()