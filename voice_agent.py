import speech_recognition as sr
import anthropic
import os
from dotenv import load_dotenv

# Загружаем API ключ
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def listen_to_voice():
    """Слушает голос и возвращает текст"""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\n🔴 Нажмите ENTER для записи...")
        input()  # Ждём нажатия Enter
        
        print("🎤 Говорите сейчас!")
        
        # Убираем фоновый шум
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            # Слушаем (до 10 секунд фразы)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("⏳ Распознаю...")
            
            # Распознаём речь (русский язык)
            text = recognizer.recognize_google(audio, language="ru-RU")
            return text
            
        except sr.WaitTimeoutError:
            print("❌ Время ожидания истекло")
            return None
        except sr.UnknownValueError:
            print("❌ Не удалось распознать речь")
            return None
        except sr.RequestError as e:
            print(f"❌ Ошибка сервиса: {e}")
            return None

def ask_claude(question):
    """Отправляет вопрос в Claude и возвращает ответ"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Ошибка API: {e}"

def main():
    """Основной цикл программы"""
    print("=" * 50)
    print("🎙️  ГОЛОСОВОЙ АГЕНТ")
    print("=" * 50)
    print("\nПримеры команд:")
    print("  - Посчитай 25 плюс 17")
    print("  - Дай определение машинного обучения")
    print("  - Скажи анекдот про программистов")
    print("\n💡 Нажимайте ENTER чтобы начать запись")
    print("💡 Ctrl+C для выхода\n")
    
    while True:
        try:
            # Слушаем голос (после нажатия Enter)
            user_text = listen_to_voice()
            
            if user_text:
                print(f"\n👤 Вы сказали: {user_text}")
                
                # Отправляем в Claude
                answer = ask_claude(user_text)
                
                # Выводим ответ
                print(f"\n🤖 Claude: {answer}\n")
                print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break

if __name__ == "__main__":
    main()