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
    print("\n" + "="*60)
    print("🔍 ШАГ 1: RAG - Поиск в документации")
    print("="*60)
    
    relevant_docs = semantic_search(user_question, knowledge_docs)
    
    print(f"Найдено документов: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. {doc['source']} (релевантность: {doc['score']:.2f})")
    
    context = "\n\n".join([f"[{doc['source']}]\n{doc['content']}" for doc in relevant_docs])
    
    system_prompt = f"""Ты — ассистент техподдержки фитнес-приложения FitTrack.

FitTrack — это приложение для отслеживания тренировок, питания и здоровья.

БАЗА ЗНАНИЙ:
{context}

ИНСТРУКЦИИ:
- Отвечай на основе документации
- Используй инструменты для проверки подписки, устройств и тикетов пользователя
- Будь дружелюбным и мотивирующим
- Если видишь открытый тикет по похожей проблеме — упомяни это
- Предлагай конкретные шаги решения, а не общие советы
"""

    user_message = user_question
    if user_id:
        user_message = f"[User ID: {user_id}]\n{user_question}"
        print(f"\n👤 Пользователь: {user_id}")
    
    messages = [{"role": "user", "content": user_message}]
    
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env={}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_list = await session.list_tools()
            
            mcp_tools = []
            for tool in tools_list.tools:
                mcp_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
            
            print("\n" + "="*60)
            print("🤖 ШАГ 2: Claude анализирует вопрос")
            print("="*60)
            print("Доступные инструменты MCP:")
            for tool in mcp_tools:
                print(f"  - {tool['name']}")
            
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            iteration = 0
            while True:
                iteration += 1
                
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=messages,
                    tools=mcp_tools
                )
                
                if response.stop_reason == "end_turn":
                    print("\n" + "="*60)
                    print("✅ ШАГ 3: Claude готов ответить")
                    print("="*60)
                    print("Источники:")
                    print("  📚 Документация (RAG)")
                    if user_id:
                        print("  💾 Данные пользователя (MCP)")
                    print("  🧠 Общие знания Claude")
                    print()
                    
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    return final_text
                
                elif response.stop_reason == "tool_use":
                    print("\n" + "="*60)
                    print(f"🔧 ШАГ 3.{iteration}: Claude вызывает MCP инструменты")
                    print("="*60)
                    
                    messages.append({"role": "assistant", "content": response.content})
                    
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"Вызов: {block.name}({block.input})")
                            
                            result = await session.call_tool(block.name, block.input)
                            result_text = result.content[0].text
                            
                            print(f"Результат: {result_text[:100]}...")
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text
                            })
                    
                    messages.append({"role": "user", "content": tool_results})
                    print("Claude получил данные, генерирует ответ...")
                else:
                    return "Ошибка обработки"

# === Инициализация ===
print("🏃 FitTrack Support Assistant")
print("=" * 50)
print("Загружаем базу знаний...")
knowledge_docs = load_knowledge_base()
knowledge_docs = create_embeddings(knowledge_docs)
print(f"✅ Документов загружено: {len(knowledge_docs)}\n")

# === Интерактивный режим ===
async def interactive_mode():
    print("Команды:")
    print("  - Введите вопрос для общего ответа")
    print("  - /user <id> <вопрос> - вопрос от конкретного пользователя")
    print("  - /users - список пользователей")
    print("  - /quit - выход\n")
    
    # Загружаем список пользователей
    with open("crm_data.json", "r", encoding="utf-8") as f:
        crm_data = json.load(f)
    
    while True:
        user_input = input("Вы: ").strip()
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            print("👋 До встречи!")
            break
        
        if user_input == "/users":
            print("\n📋 Зарегистрированные пользователи:")
            for uid, user in crm_data["users"].items():
                print(f"  {uid}: {user['name']} ({user['email']}) - {user['subscription']}")
            print()
            continue
        
        # Команда /user
        if user_input.startswith("/user "):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                print("❌ Формат: /user <user_id> <вопрос>\n")
                continue
            user_id = parts[1]
            question = parts[2]
            
            if user_id not in crm_data["users"]:
                print(f"❌ Пользователь {user_id} не найден. Используйте /users для списка\n")
                continue
            
            print(f"\n🤖 Ассистент (для {crm_data['users'][user_id]['name']}):")
            answer = await support_agent(question, user_id=user_id)
            print(answer)
            print()
        else:
            # Обычный вопрос
            print("\n🤖 Ассистент:")
            answer = await support_agent(user_input)
            print(answer)
            print()

if __name__ == "__main__":
    asyncio.run(interactive_mode())