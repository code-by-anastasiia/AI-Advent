# Literary Critic AI - VPS Deployment

AI-powered literary critic chatbot deployed on VPS with local LLM.

![Demo](screenshot.png) <!-- можешь добавить скриншот чата -->

## 🚀 Live Demo

**Web Interface:** [https://твой-username.github.io/название-репо/](ссылка_появится_после_настройки_Pages)

**API Endpoint:** http://46.224.229.113:8000

## 🛠️ Tech Stack

- **Frontend:** HTML/CSS/JavaScript (vanilla)
- **Backend:** FastAPI
- **LLM:** Ollama + gemma2:2b (1.6GB)
- **Hosting:** Hetzner Cloud VPS (Ubuntu 24.04)
- **Server:** Uvicorn

## 📋 Features

- 💬 Real-time chat interface
- 🎨 Beautiful gradient UI with animations
- 📚 Literary knowledge and analysis
- 💾 Conversation history per user
- 🌐 Accessible from anywhere

## 🔧 Installation on VPS

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Model
```bash
ollama pull gemma2:2b
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 4. Run Server
```bash
nohup python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

## 📡 API Endpoints

### POST /chat
Send message to the bot
```json
{
  "user_id": "user123",
  "message": "Что ты думаешь о Достоевском?"
}
```

### GET /health
Check server status
```json
{
  "status": "ok",
  "model": "gemma2:2b"
}
```

### DELETE /history/{user_id}
Clear user conversation history

## 🌐 API Documentation

Interactive API docs: http://46.224.229.113:8000/docs

## 💻 Local Development
```bash
python test_remote.py
```

## 📊 VPS Specifications

- **Provider:** Hetzner Cloud
- **Plan:** CX23 (Cost-Optimized)
- **CPU:** 2 vCPU (x86)
- **RAM:** 4 GB
- **Storage:** 40 GB SSD
- **Cost:** €3.56/month

## 🎓 Course Project

Day 27: Local LLM on VPS - AI Agents Course

## 📝 License

MIT
