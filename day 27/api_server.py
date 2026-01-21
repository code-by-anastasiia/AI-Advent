from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import json
import os

app = FastAPI()

# Папка для историй пользователей
HISTORY_DIR = "chat_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

# Модель для входящих запросов
class ChatRequest(BaseModel):
    user_id: str  # ID пользователя (чтобы у каждого была своя история)
    message: str  # Сообщение пользователя

# Модель для ответов
class ChatResponse(BaseModel):
    response: str
    
# Функции для работы с историей (как в твоём боте)
def get_history_file(user_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{user_id}.json")

def load_history(user_id: str) -> list:
    file_path = get_history_file(user_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(user_id: str, history: list):
    file_path = get_history_file(user_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Отправить сообщение литературному критику"""
    try:
        # Загружаем историю этого пользователя
        history = load_history(request.user_id)
        
        # Добавляем новое сообщение
        history.append({
            "role": "user",
            "content": request.message
        })
        
        # Получаем ответ от Ollama
        response = ollama.chat(
            model='gemma3:4b',
            messages=history
        )
        
        assistant_message = response['message']['content']
        
        # Сохраняем ответ в историю
        history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        save_history(request.user_id, history)
        
        return ChatResponse(response=assistant_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Очистить историю пользователя"""
    file_path = get_history_file(user_id)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "cleared"}
    return {"status": "no_history"}

@app.get("/health")
async def health():
    """Проверка работоспособности сервера"""
    return {"status": "ok", "model": "gemma3:4b"}