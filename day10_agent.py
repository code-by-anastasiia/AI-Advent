from anthropic import Anthropic
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MEMORY_FILE = "memory.json"
COMPRESSION_THRESHOLD = 10  # Сжимать каждые 10 сообщений

# === ФУНКЦИИ ПАМЯТИ ===

def load_memory():
    """Загрузить историю из файла"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Загружено {len(data)} сообщений из памяти")
            return data
    print("Память пуста, начинаем новый разговор")
    return []

def save_memory(history):
    """Сохранить историю в файл"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Сохранено в файл")

def add_message(history, role, content):
    """Добавить сообщение"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history.append(message)
    return history

def clear_memory():
    """Очистить память"""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        print("Память очищена")
    else:
        print("Память уже пуста")

# === ФУНКЦИИ СЖАТИЯ ===

def create_summary(messages):
    """Создать summary для списка сообщений"""
    # Формируем текст диалога
    dialog = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in messages
        if '[SUMMARY]' not in msg.get('content', '')
    ])
    
    # Просим Claude сделать краткое содержание
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Сделай краткое содержание (2-3 предложения):\n\n{dialog}"
        }]
    )
    
    return response.content[0].text

def compress_history(history):
    """Сжать историю каждые N сообщений"""
    # Считаем обычные сообщения (не summary)
    regular_messages = [
        msg for msg in history 
        if '[SUMMARY]' not in msg.get('content', '')
    ]
    
    if len(regular_messages) < COMPRESSION_THRESHOLD:
        return history  # Ещё рано сжимать
    
    print(f"\nСЖАТИЕ (обычных сообщений: {len(regular_messages)})")
    
    # Берём первые N обычных сообщений
    to_compress = regular_messages[:COMPRESSION_THRESHOLD]
    
    # Создаём summary
    summary_text = create_summary(to_compress)
    summary_msg = {
        "role": "assistant",
        "content": f"[SUMMARY]: {summary_text}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Создаём новую историю:
    # 1. Все старые summary
    # 2. Новый summary
    # 3. Несжатые сообщения
    new_history = []
    compressed_count = 0
    
    for msg in history:
        if '[SUMMARY]' in msg.get('content', ''):
            new_history.append(msg)  # Оставляем старые summary
        elif compressed_count < COMPRESSION_THRESHOLD:
            compressed_count += 1  # Пропускаем (сжимаем)
        else:
            new_history.append(msg)  # Оставляем новые
    
    # Добавляем новый summary
    new_history.insert(0, summary_msg)
    
    print(f"Сжато: {len(history)} → {len(new_history)} сообщений")
    
    return new_history

# === ФУНКЦИИ ЧАТА ===

def send_message(user_text, history):
    """Отправить сообщение"""
    # Добавляем сообщение пользователя
    history = add_message(history, "user", user_text)
    
    # Готовим для API (без timestamp)
    api_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
    ]
    
    # Отправляем
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=api_history
    )
    
    # Сохраняем ответ
    ai_reply = response.content[0].text
    history = add_message(history, "assistant", ai_reply)
    
    # Проверяем: нужно ли сжатие?
    history = compress_history(history)
    
    # Сохраняем в файл
    save_memory(history)
    
    return ai_reply, history

def show_history(history):
    """Показать всю историю"""
    if not history:
        print("История пуста")
        return
    
    print(f"\n ИСТОРИЯ ({len(history)} сообщений):")
    print("="*70)
    for i, msg in enumerate(history, 1):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        timestamp = msg.get("timestamp", "неизвестно")
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        
        if '[SUMMARY]' in content:
            print(f"{i}. [{timestamp}] {content}")
        else:
            print(f"{i}. [{timestamp}] {role_icon} {content}")
    print("="*70)

# === ГЛАВНАЯ ПРОГРАММА ===

def main():
    print("="*70)
    print("АГЕНТ С ПАМЯТЬЮ И СЖАТИЕМ")
    print(f"   Сжатие каждые {COMPRESSION_THRESHOLD} сообщений")
    print("="*70)
    
    # Загружаем память при запуске
    history = load_memory()
    
    print("\nКоманды:")
    print("  'выход' - выйти")
    print("  'история' - показать всю историю")
    print("  'очистить' - удалить всю память")
    print("="*70)
    
    while True:
        user_input = input("\nВы: ").strip()
        
        if not user_input:
            continue
        
        # Команды
        if user_input.lower() == "выход":
            print(f"\nВ памяти: {len(history)} сообщений")
            print("До свидания!")
            break
        
        if user_input.lower() == "история":
            show_history(history)
            continue
        
        if user_input.lower() == "очистить":
            confirm = input("Удалить всю память? (да/нет): ")
            if confirm.lower() == "да":
                clear_memory()
                history = []
            continue
        
        # Обычное сообщение
        try:
            reply, history = send_message(user_input, history)
            print(f"\nAI: {reply}")
            
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()