# day25_local_llm.py
import requests

def ask_local_llm(question, model="gemma3:4b"):
    """Отправить вопрос локальной модели через Ollama API"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": question,
            "stream": False  # получить весь ответ сразу
        }
    )
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Ошибка: {response.status_code}"


if __name__ == "__main__":
    # Тест 1: вопрос на английском
    print("Вопрос 1: What is machine learning?")
    answer1 = ask_local_llm("What is machine learning?")
    print(f"Ответ: {answer1}\n")
    print("="*60)
    
    # Тест 2: вопрос на русском
    print("\nВопрос 2: Что такое рекурсия в программировании?")
    answer2 = ask_local_llm("Что такое рекурсия в программировании?")
    print(f"Ответ: {answer2}\n")
    print("="*60)
    
    # Тест 3: код
    print("\nВопрос 3: Write a Python function to check if number is prime")
    answer3 = ask_local_llm("Write a Python function to check if number is prime")
    print(f"Ответ:\n{answer3}")