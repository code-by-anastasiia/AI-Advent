# literary_critic_optimized.py
import requests
import time

class LocalLLMChat:
    def __init__(self, model_name="gemma3:4b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
        
        # Оптимизированные параметры (можно менять)
        self.params = {
            "temperature": 0.7,      # баланс креативность/точность
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 400,      # средняя длина ответа
            "num_ctx": 2048,
        }
        
        # Улучшенный промпт - конкретнее и структурированнее
        self.system_prompt = """Ты редактор издательства. Твоя задача - подготовить текст к публикации.

Проверь:
1. ЯЗЫК: штампы, повторы, канцеляризмы (процитируй проблемные места)
2. РИТМ: длина предложений, монотонность (покажи где "спотыкаешься")
3. ОБРАЗЫ: клише vs свежие метафоры (приведи примеры)

Формат ответа:
✓ Оставить: [что хорошо]
✗ Убрать/изменить: "[цитата]" → предложи замену
! Главное замечание: [один совет]

Будь конкретным. Каждое замечание = цитата + решение."""
    
    def send_message(self, user_message):
        # Формируем промпт с историей
        prompt = f"System: {self.system_prompt}\n\n"
        
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += f"User: {user_message}\nAssistant:"
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self.params
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            assistant_message = result["response"]
            duration = time.time() - start_time
            
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return {
                "text": assistant_message,
                "duration": duration,
                "tokens": result.get("eval_count", 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {"text": f"Ошибка: {e}", "duration": 0, "tokens": 0}
    
    def set_params(self, **kwargs):
        """Изменить параметры на лету"""
        self.params.update(kwargs)
        print(f"✓ Обновлено: {kwargs}")
    
    def clear_history(self):
        self.conversation_history = []
        print("✓ История очищена")

def main():
    print("=" * 60)
    print("📚 Литературный Критик AI (Оптимизированный)")
    print("=" * 60)
    print("Команды:")
    print("  /params - показать параметры")
    print("  /temp <0.0-1.0> - изменить креативность")
    print("  /clear - очистить историю")
    print("  /exit - выход")
    print("=" * 60)
    
    chat = LocalLLMChat()
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if not user_input:
            continue
            
        if user_input == "/exit":
            print("До свидания! 👋")
            break
        
        if user_input == "/clear":
            chat.clear_history()
            continue
        
        if user_input == "/params":
            print("\nТекущие параметры:")
            for k, v in chat.params.items():
                print(f"  {k}: {v}")
            continue
        
        if user_input.startswith("/temp "):
            try:
                temp = float(user_input.split()[1])
                chat.set_params(temperature=temp)
            except:
                print("❌ Формат: /temp 0.7")
            continue
        
        print("📚 Критик: ", end="", flush=True)
        result = chat.send_message(user_input)
        print(result["text"])
        print(f"\n[{result['duration']:.1f}с | {result['tokens']} токенов]")

if __name__ == "__main__":
    main()