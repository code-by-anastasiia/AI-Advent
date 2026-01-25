import ollama
import json

class WeatherAnalyst:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.data = None
        self.data_text = ""
    
    def load_data(self, filepath):
        """Загружаем JSON с погодными данными"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Превращаем данные в читаемый текст
        self.data_text = json.dumps(self.data, indent=2, ensure_ascii=False)
        
        print(f"✓ Загружено {len(self.data)} записей о погоде")
        print(f"✓ Период: {self.data[0]['дата']} - {self.data[-1]['дата']}")
    
    def analyze(self, question):
        """Задаём аналитический вопрос"""
        prompt = f"""Ты — аналитик погодных данных. Вот данные за период:

{self.data_text}

Вопрос: {question}

Проанализируй данные и дай точный ответ. Если нужно посчитать, считай внимательно."""

        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content']


# Использование
if __name__ == "__main__":
    analyst = WeatherAnalyst()
    analyst.load_data('weather.json')
    
    print("\n" + "="*50)
    print("ЛОКАЛЬНЫЙ АНАЛИТИК ПОГОДЫ")
    print("="*50 + "\n")
    
    # Примеры вопросов
    questions = [
        "Какая была самая низкая температура?",
        "Сколько дней шёл снег?",
        "В какой день был самый сильный ветер?",
        "Какая средняя температура за период?"
    ]
    
    for q in questions:
        print(f"❓ {q}")
        answer = analyst.analyze(q)
        print(f"💡 {answer}\n")