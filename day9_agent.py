from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

history = []
total_tokens = 0

def compress_history():
    global history
    
    # СТАТИСТИКА ДО
    messages_before = len(history)
    tokens_before = sum(len(m['content']) for m in history) // 4
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ДО СЖАТИЯ:")
    print(f"   Сообщений в памяти: {messages_before}")
    print(f"   ~Токенов в истории: {tokens_before}")
    print("="*60)
    
    # СЖАТИЕ
    old_messages = history[:10]
    dialog = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": f"Краткое содержание (2-3 предложения):\n\n{dialog}"}]
    )
    
    summary = response.content[0].text
    history = [{"role": "assistant", "content": f"[SUMMARY]: {summary}"}] + history[10:]
    
    # СТАТИСТИКА ПОСЛЕ
    messages_after = len(history)
    tokens_after = sum(len(m['content']) for m in history) // 4
    
    print("\n" + "="*60)
    print("СТАТИСТИКА ПОСЛЕ СЖАТИЯ:")
    print(f"   Сообщений в памяти: {messages_after}")
    print(f"   ~Токенов в истории: {tokens_after}")
    print("-"*60)
    print(f"ЭКОНОМИЯ:")
    print(f"   Сообщений удалено: {messages_before - messages_after}")
    print(f"   ~Токенов сэкономлено: {tokens_before - tokens_after}")
    print(f"   Процент экономии: {((tokens_before - tokens_after) / tokens_before * 100):.1f}%")
    print("="*60)

def check_and_compress():
    if len(history) >= 10:
        print("\nНАЧИНАЮ СЖАТИЕ...")
        compress_history()

def send_message(user_text):
    global total_tokens
    
    history.append({"role": "user", "content": user_text})
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=history
    )
    
    ai_reply = response.content[0].text
    history.append({"role": "assistant", "content": ai_reply})
    
    tokens = response.usage.input_tokens + response.usage.output_tokens
    total_tokens += tokens
    
    print(f"\nТокены: {tokens} | Всего: {total_tokens} | Сообщений: {len(history)}")
    
    return ai_reply

# ОСНОВНОЙ ЦИКЛ
print("="*60)
print("ЧАТ СО СЖАТИЕМ ДИАЛОГА")
print("   Сжатие каждые 10 сообщений")
print("   Команды: 'выход', 'история'")
print("="*60)

while True:
    user_input = input("\nВы: ").strip()
    
    if not user_input:
        continue
    
    if user_input == "выход":
        print("\n" + "="*60)
        print("ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Всего токенов: {total_tokens}")
        print(f"   Сообщений в памяти: {len(history)}")
        print("="*60)
        break
    
    if user_input == "история":
        print(f"\nИстория ({len(history)} сообщений):")
        for i, msg in enumerate(history, 1):
            content = msg['content'][:70] + "..." if len(msg['content']) > 70 else msg['content']
            print(f"{i}. {msg['role']}: {content}")
        continue
    
    reply = send_message(user_input)
    print(f"AI: {reply}")
    
    check_and_compress()

print("\nГотово!")