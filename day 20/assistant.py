import os
import streamlit as st
from pathlib import Path
from anthropic import Anthropic
from project_rag import ProjectRAG
from dotenv import load_dotenv

load_dotenv()


class DeveloperAssistant:
    """
    Ассистент разработчика с RAG (без git)
    """
    
    def __init__(self, work_dir):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.work_dir = Path(work_dir)
        self.rag = ProjectRAG(work_dir)
        
    def initialize(self):
        """
        Индексируем проект
        """
        self.rag.index_project()
    
    def get_git_context(self):
        """
        Заглушка для git
        """
        return "Git не используется в этом проекте"
    
    def handle_help(self, question):
        """
        Обрабатываем команду /help
        """
        # Получаем контекст из документации
        docs_context = self.rag.search(question, top_k=3)
        
        # Получаем git-контекст (заглушка)
        git_context = self.get_git_context()
        
        # Формируем системный промпт
        system_prompt = f"""Ты - ассистент разработчика для проекта.

Рабочая директория: {self.work_dir}

ДОКУМЕНТАЦИЯ ПРОЕКТА:
{docs_context}

Отвечай кратко и по существу на основе предоставленной документации.
Если можешь показать пример кода - покажи.
Если нужной информации нет в документации - так и скажи."""
        
        # Запрос к Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": question
            }]
        )
        
        return response.content[0].text


# === Streamlit UI ===

def main():
    st.set_page_config(page_title="Dev Assistant")
    
    st.title("Ассистент разработчика")    
    # Сайдбар с настройками
    with st.sidebar:
        st.subheader("⚙️ Настройки")
        
        work_dir = st.text_input(
            "Путь к проекту",
            value="./example_project",
            help="Папка с вашим проектом"
        )
        
        if st.button("🔄 Переиндексировать проект"):
            with st.spinner("Индексирую..."):
                assistant = DeveloperAssistant(work_dir)
                assistant.initialize()
                st.success("✓ Проект проиндексирован")
        
        st.divider()
        
        st.subheader("💡 Как использовать")
        st.markdown("""
**Команда /help:**
```
/help как работает функция X?
/help где находится код для Y?
/help покажи пример использования
```

**Без команды** - обычный ответ
        """)
    
    # Инициализация ассистента
    if "assistant" not in st.session_state:
        st.session_state.assistant = DeveloperAssistant(work_dir)
        st.session_state.assistant.initialize()
    
    # История сообщений
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Показываем историю
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Ввод пользователя
    if prompt := st.chat_input("Спроси о проекте или используй /help"):
        # Добавляем сообщение пользователя
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Обрабатываем команду
        with st.chat_message("assistant"):
            if prompt.strip().startswith("/help"):
                # Убираем /help из запроса
                question = prompt.replace("/help", "").strip()
                
                if not question:
                    response = "Задай вопрос после команды /help. Например: `/help как работает функция predict?`"
                else:
                    with st.spinner("Ищу в документации..."):
                        response = st.session_state.assistant.handle_help(question)
            else:
                response = "Используй команду `/help <вопрос>` для вопросов о проекте"
            
            st.markdown(response)
        
        # Сохраняем ответ
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

if __name__ == "__main__":
    main()