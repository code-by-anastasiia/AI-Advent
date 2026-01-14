import anthropic
import os
from pathlib import Path
import ollama
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()

# === RAG ===
def load_knowledge_base():
    kb_path = Path("knowledge_base")
    docs = []
    for file in kb_path.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({"source": file.name, "content": f.read()})
    return docs

def create_embeddings(docs):
    for doc in docs:
        response = ollama.embeddings(model="nomic-embed-text", prompt=doc["content"])
        doc["embedding"] = response["embedding"]
    return docs

def semantic_search(query, docs, top_k=2):
    import numpy as np
    query_emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    for doc in docs:
        doc["score"] = cosine_similarity(query_emb, doc["embedding"])
    
    return sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]

# === Agent ===
async def support_agent(user_question, user_id=None):
    # 1. RAG
    relevant_docs = semantic_search(user_question, knowledge_docs)
    context = "\n\n".join([f"[{doc['source']}]\n{doc['content']}" for doc in relevant_docs])
    
    # 2. Системный промпт
    system_prompt = f"""Ты — ассистент техподдержки.

БАЗА ЗНАНИЙ:
{context}

ИНСТРУКЦИИ:
- Отвечай на основе документации
- Используй инструменты для проверки данных пользователя/тикетов
- Будь конкретным и полезным
"""

    # 3. Формируем сообщение
    user_message = user_question
    if user_id:
        user_message = f"[User ID: {user_id}]\n{user_question}"
    
    messages = [{"role": "user", "content": user_message}]
    
    # 4. Подключаемся к MCP и держим сессию открытой
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env={}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_list = await session.list_tools()
            
            # Конвертируем MCP tools в формат Anthropic
            mcp_tools = []
            for tool in tools_list.tools:
                mcp_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
            
            # 5. Agentic loop
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=messages,
                    tools=mcp_tools
                )
                
                # Проверяем stop_reason
                if response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    return final_text
                
                elif response.stop_reason == "tool_use":
                    # Добавляем ответ ассистента
                    messages.append({"role": "assistant", "content": response.content})
                    
                    # Вызываем тулы
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            # Вызываем через сессию внутри контекста
                            result = await session.call_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.content[0].text
                            })
                    
                    # Добавляем результаты
                    messages.append({"role": "user", "content": tool_results})
                else:
                    return "Ошибка обработки"

# === Инициализация ===
print("Загружаем базу знаний...")
knowledge_docs = load_knowledge_base()
knowledge_docs = create_embeddings(knowledge_docs)
print(f"Документов: {len(knowledge_docs)}\n")

# === Тест ===
async def main():
    print("=== Тест 1: Общий вопрос ===")
    answer = await support_agent("Как восстановить пароль?")
    print(answer)
    
    print("\n=== Тест 2: С контекстом пользователя ===")
    answer = await support_agent("Почему не работает авторизация?", user_id="user123")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())