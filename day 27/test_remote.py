import requests

# IP твоего VPS
SERVER_IP = "46.224.229.113"

# Тест 1: Проверка здоровья
print("Тестируем удалённый сервер...")
response = requests.get(f"http://{SERVER_IP}:8000/health")
print("Health check:", response.json())

# Тест 2: Отправка сообщения
print("\nОтправляем вопрос боту...")
response = requests.post(
    f"http://{SERVER_IP}:8000/chat",
    json={
        "user_id": "miranda",
        "message": "Привет! Что ты думаешь о Пушкине?"
    }
)

# Посмотрим, что пришло
print("\nПолный ответ:")
print(response.json())