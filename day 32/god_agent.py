import os
from anthropic import Anthropic
import ollama
from dotenv import load_dotenv
from tools import ToolsManager

load_dotenv()

class GodAgent:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY не найден!")
        
        self.client = Anthropic(api_key=api_key)
        self.tools_manager = ToolsManager()
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge()
        
        print(f"📚 База знаний: {len(self.knowledge_base)} документов")
    
    def load_knowledge(self):
        """Загрузка документации"""
        docs = []
        kb_path = "knowledge_base"
        
        if os.path.exists(kb_path):
            for file in os.listdir(kb_path):
                if file.endswith(('.md', '.txt')):
                    with open(f"{kb_path}/{file}", 'r', encoding='utf-8') as f:
                        docs.append({'file': file, 'content': f.read()})
        
        return docs
    
    def search_knowledge(self, query):
        """RAG поиск через embeddings"""
        if not self.knowledge_base:
            return None
        
        try:
            query_emb = ollama.embeddings(
                model='nomic-embed-text',
                prompt=query
            )['embedding']
            
            best_doc = None
            best_score = -1
            
            for doc in self.knowledge_base:
                doc_emb = ollama.embeddings(
                    model='nomic-embed-text',
                    prompt=doc['content'][:500]
                )['embedding']
                
                score = sum(a*b for a,b in zip(query_emb, doc_emb))
                if score > best_score:
                    best_score = score
                    best_doc = doc
            
            return best_doc if best_score > 0.35 else None
        except:
            return None
    
    def get_tools_definition(self):
        """Описание всех инструментов для Claude"""
        return [
            {
                "name": "add_task",
                "description": "Добавить задачу в список дел",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        "deadline": {"type": "string"}
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "complete_task",
                "description": "Отметить задачу как выполненную",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "get_tasks",
                "description": "Получить список задач",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["pending", "completed", "all"]}
                    }
                }
            },
            {
                "name": "add_note",
                "description": "Создать заметку",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "search_notes",
                "description": "Найти заметки",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_weather",
                "description": "Узнать погоду в городе",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            },
            {
                "name": "search_docs",
                "description": "Искать в базе знаний (документация проектов)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def execute_tool(self, tool_name, tool_input):
        """Выполнение инструмента"""
        if tool_name == "add_task":
            return self.tools_manager.add_task(
                tool_input['title'],
                tool_input.get('priority', 'medium'),
                tool_input.get('deadline')
            )
        elif tool_name == "complete_task":
            return self.tools_manager.complete_task(tool_input['task_id'])
        elif tool_name == "get_tasks":
            return self.tools_manager.get_tasks(tool_input.get('status', 'all'))
        elif tool_name == "add_note":
            return self.tools_manager.add_note(
                tool_input['title'],
                tool_input['content'],
                tool_input.get('tags')
            )
        elif tool_name == "search_notes":
            return self.tools_manager.search_notes(tool_input['query'])
        elif tool_name == "get_weather":
            return self.tools_manager.get_weather(tool_input.get('city', 'Neumünster'))
        elif tool_name == "search_docs":
            doc = self.search_knowledge(tool_input['query'])
            if doc:
                return f"📄 {doc['file']}:\n\n{doc['content'][:600]}..."
            return "🔍 Ничего не найдено"
    
    def chat(self, user_message):
        """Диалог с агентом"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            tools=self.get_tools_definition(),
            messages=self.conversation_history
        )
        
        # Tool use цикл
        while response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 {block.name}")
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                tools=self.get_tools_definition(),
                messages=self.conversation_history
            )
        
        # Извлечь ответ
        assistant_message = ""
        for block in response.content:
            if hasattr(block, "text"):
                assistant_message += block.text
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message


def main():
    print("=" * 60)
    print("🤖 GOD AGENT - Персональный ассистент")
    print("=" * 60)
    print("\n💡 Умею:")
    print("  • Управлять задачами и заметками")
    print("  • Искать в базе знаний")
    print("  • Показывать погоду")
    print("\n📝 Команды: /exit, /clear\n")
    
    try:
        agent = GodAgent()
    except Exception as e:
        print(f"❌ {e}")
        return
    
    while True:
        try:
            user_input = input("💬 ").strip()
            
            if user_input == "/exit":
                print("\n👋 Пока!")
                break
            
            if user_input == "/clear":
                agent.conversation_history = []
                print("🗑️ История очищена\n")
                continue
            
            if not user_input:
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Пока!")
            break
        except Exception as e:
            print(f"\n❌ {e}\n")


if __name__ == "__main__":
    main()