from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Загрузка API ключа
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Ошибка: API ключ не найден в .env файле!")
    exit(1)

client = Anthropic(api_key=api_key)

print("=" * 80)
print("ТЕСТЕР TEMPERATURE")
print("=" * 80)

# Пользователь вводит свой промпт
user_prompt = input("\nВведите ваш промпт:\n> ")

if not user_prompt.strip():
    print("Ошибка: Промпт не может быть пустым!")
    exit(1)

# Стандартные значения temperature
temperatures = [0, 0.5, 1.0]

print("\n" + "=" * 80)
print(f"Temperature: {temperatures}")
print("=" * 80)

# Основное тестирование
for temp in temperatures:
    print(f"\nTEMPERATURE = {temp}")
    print("-" * 80)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=temp,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.content[0].text
        print(f"\n{answer}\n")
        
    except Exception as e:
        print(f"Ошибка: {e}\n")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("Готово!")
print("=" * 80)