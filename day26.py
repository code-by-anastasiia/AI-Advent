import requests
import json

class LocalLLMChat:
    def __init__(self, model_name="gemma3:4b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
    
    def send_message(self, user_message):
        """Отправляем сообщение модели и получаем ответ"""
        
        # System prompt - литературный критик
        system_prompt = """Ты профессиональный литературный критик. 
Анализируй тексты с точки зрения стиля, структуры, образности и художественных приёмов. 
Давай конструктивную критику: указывай на сильные стороны и предлагай улучшения. 
Будь объективным, но доброжелательным."""
        
        # Формируем промпт с системным сообщением и историей
        prompt = f"System: {system_prompt}\n\n"
        
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += f"User: {user_message}\nAssistant:"
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            assistant_message = result["response"]
            
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            return f"Ошибка связи с моделью: {e}"
    
    def clear_history(self):
        """Очищаем историю диалога"""
        self.conversation_history = []
        print("История диалога очищена.")

def main():
    """Основная функция - запуск CLI-чата"""
    print("=" * 60)
    print("📚 Литературный Критик AI")
    print("=" * 60)
    print("Команды:")
    print("  /clear - очистить историю диалога")
    print("  /exit или /quit - выйти из программы")
    print("=" * 60)
    print()
    
    chat = LocalLLMChat(model_name="gemma3:4b")
    
    while True:
        user_input = input("Вы: ").strip()
        
        if user_input.lower() in ["/exit", "/quit"]:
            print("До свидания! 👋")
            break
        
        if user_input.lower() == "/clear":
            chat.clear_history()
            continue
        
        if not user_input:
            continue
        
        print("📚 Критик: ", end="", flush=True)
        response = chat.send_message(user_input)
        print(response)
        print()

if __name__ == "__main__":
    main()