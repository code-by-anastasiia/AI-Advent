# День 8. Работа с токенами

# - Добавьте в код подсчёт токенов (на запрос и ответ)
# - Сравните: короткий запрос, длинный запрос и запрос, превышающий лимит модели

from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("=" * 80)
print("СЧЁТЧИК ТОКЕНОВ")
print("=" * 80)

while True:
    print("\nВыберите:")
    print("1 - Ввести запрос")
    print("2 - Тест превышения лимита")
    print("3 - Выход")
    
    choice = input("> ")
    
    if choice == "3":
        break
    
    elif choice == "2":
        print("\nТЕСТ ПРЕВЫШЕНИЯ ЛИМИТА")
        huge_text = "Повтори эту фразу. " * 100000
        print(f"Символов: {len(huge_text):,}")
        print(f"Примерно токенов: ~{len(huge_text) // 2.5:,.0f}")
        print("Отправка...\n")
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": huge_text}]
            )
            print(f"Прошёл! Токенов: {response.usage.input_tokens:,}")
        except Exception as e:
            print(f"ОШИБКА: {e}")
    
    elif choice == "1":
        user_prompt = input("\nВведите запрос: ")
        
        if not user_prompt.strip():
            continue
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            print("\n" + "-" * 80)
            print(f"Символов: {len(user_prompt)}")
            print(f"Токенов в запросе: {response.usage.input_tokens}")
            print(f"Токенов в ответе: {response.usage.output_tokens}")
            print(f"Всего: {response.usage.input_tokens + response.usage.output_tokens}")
            print("-" * 80)
            print(f"\nОтвет:\n{response.content[0].text}\n")
            
        except Exception as e:
            print(f"Ошибка: {e}")

print("\nПрограмма завершена!")
