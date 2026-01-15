#!/usr/bin/env python3
"""
Командный ассистент для управления задачами курса AI агентов.
Комбинирует RAG (документация) и MCP (управление задачами).
"""

import os
import sys
import json
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rag_system import ProjectRAG
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class TeamAssistant:
    def __init__(self):
        """Инициализация командного ассистента"""
        
        # Anthropic клиент
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY не найден в .env файле")
        
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
        
        # RAG система
        print("[ASSISTANT] Инициализация RAG...", file=sys.stderr)
        self.rag = ProjectRAG("knowledge_base")
        self.rag.index_documents()
        
        # MCP сессия (будет инициализирована при запуске)
        self.mcp_session = None
        self.mcp_tools = []
        self.stdio_context = None
        
        print("[ASSISTANT] Готов к работе!", file=sys.stderr)
    
    async def init_mcp(self):
        """Инициализация MCP клиента"""
        print("[ASSISTANT] Подключение к MCP серверу...", file=sys.stderr)
        
        # Параметры запуска MCP сервера
        server_params = StdioServerParameters(
            command="python",
            args=["task_server.py"],
            env=None
        )
        
        # Создаем async context manager
        self.stdio_context = stdio_client(server_params)
        stdio_transport = await self.stdio_context.__aenter__()
        read_stream, write_stream = stdio_transport
        
        # Создаем сессию
        self.mcp_session = ClientSession(read_stream, write_stream)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()
        
        # Получаем список tools
        response = await self.mcp_session.list_tools()
        self.mcp_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]
        
        print(f"[ASSISTANT] MCP подключен: {len(self.mcp_tools)} инструментов", file=sys.stderr)
    
    async def cleanup(self):
        """Закрытие MCP соединения"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)
    
    async def process_query(self, user_query: str):
        """
        Обработка запроса пользователя
        
        Args:
            user_query: Вопрос пользователя
            
        Returns:
            Ответ ассистента
        """
        print(f"\n[USER] {user_query}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        # Шаг 1: RAG поиск
        print("[ASSISTANT] Поиск в документации...", file=sys.stderr)
        rag_results = self.rag.search(user_query, top_k=2, threshold=0.3)
        
        # Формируем контекст из RAG
        rag_context = ""
        if rag_results:
            print(f"[RAG] Найдено документов: {len(rag_results)}", file=sys.stderr)
            rag_context = "# Релевантная документация:\n\n"
            for r in rag_results:
                rag_context += f"## {r['filename']} (релевантность: {r['score']})\n"
                rag_context += f"{r['content']}\n\n"
        else:
            print("[RAG] Релевантные документы не найдены", file=sys.stderr)
        
        # Шаг 2: Формируем системный промпт
        system_prompt = f"""Ты - командный ассистент для курса AI агентов.

Твои возможности:
1. Отвечать на вопросы о проекте (используя документацию)
2. Управлять задачами через MCP tools
3. Давать рекомендации по приоритизации

{rag_context}

Когда отвечаешь:
- Если нашел инфо в документации - используй её
- Для действий с задачами - используй MCP tools
- Для приоритизации - анализируй задачи и давай конкретные рекомендации
- Отвечай по-русски, четко и по существу
"""
        
        # Шаг 3: Вызываем Claude с MCP tools
        messages = [{"role": "user", "content": user_query}]
        
        return await self._call_claude(messages, system_prompt)
    
    async def _call_claude(self, messages: list, system_prompt: str):
        """
        Вызов Claude API с возможностью использования MCP tools
        
        Args:
            messages: История сообщений
            system_prompt: Системный промпт с RAG контекстом
            
        Returns:
            Финальный ответ Claude
        """
        
        while True:
            # Вызываем Claude
            print("[ASSISTANT] Вызов Claude API...", file=sys.stderr)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt,
                messages=messages,
                tools=self.mcp_tools if self.mcp_tools else None
            )
            
            # Проверяем, нужно ли использовать tool
            if response.stop_reason == "tool_use":
                # Claude хочет вызвать MCP tool
                tool_use_block = next(
                    (block for block in response.content if block.type == "tool_use"),
                    None
                )
                
                if tool_use_block:
                    tool_name = tool_use_block.name
                    tool_input = tool_use_block.input
                    
                    print(f"[MCP] Claude вызывает tool: {tool_name}", file=sys.stderr)
                    print(f"[MCP] Параметры: {json.dumps(tool_input, ensure_ascii=False)}", file=sys.stderr)
                    
                    # Вызываем MCP tool
                    tool_result = await self.mcp_session.call_tool(tool_name, tool_input)
                    
                    # Извлекаем текст из результата
                    result_text = ""
                    for content in tool_result.content:
                        if hasattr(content, 'text'):
                            result_text += content.text
                    
                    print(f"[MCP] Результат получен", file=sys.stderr)
                    
                    # Добавляем ответ assistant и результат tool в историю
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "content": result_text
                            }
                        ]
                    })
                    
                    # Продолжаем цикл - Claude обработает результат
                    continue
            
            # Если tool_use не нужен - возвращаем финальный ответ
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text
            
            print("[ASSISTANT] Ответ готов", file=sys.stderr)
            return final_text

async def main():
    """Главная функция - интерактивный режим"""
    
    print("\n" + "=" * 60)
    print("Командный ассистент для курса AI агентов")
    print("=" * 60)
    print("\nИнициализация...")
    
    # Создаем ассистента
    assistant = TeamAssistant()
    
    # Инициализируем MCP
    await assistant.init_mcp()
    
    print("\n✓ Готов к работе!")
    print("\nПримеры команд:")
    print("  - Покажи задачи с приоритетом high")
    print("  - Что такое RAG?")
    print("  - Создай задачу: добавить тесты")
    print("  - Какие задачи с багами?")
    print("  - Что я делала на День 22?")
    print("\nВведите 'exit' для выхода\n")
    
    # Интерактивный цикл
    try:
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'выход']:
                    print("\nДо встречи!")
                    break
                
                # Обрабатываем запрос
                response = await assistant.process_query(user_input)
                
                # Выводим ответ
                print("\n" + "=" * 60)
                print(response)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\nПрервано пользователем. До встречи!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
    
    finally:
        # Закрываем MCP соединение
        await assistant.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())