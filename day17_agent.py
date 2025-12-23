from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import json

# ===== ЗАГРУЗКА API КЛЮЧА =====
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Ошибка: API ключ не найден!")
    exit(1)

class ClaudeRAGAgent:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Инициализация RAG-агента с Claude
        
        Args:
            api_key: Ключ для Anthropic
            model: Название модели Claude (haiku, sonnet, opus)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ не найден. Укажите явно или установите ANTHROPIC_API_KEY")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        
        # Модель для эмбеддингов
        print("Загрузка модели для эмбеддингов...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Модель для эмбеддингов загружена")
        
        # Инициализация ChromaDB с новым API
        print("Инициализация векторной базы данных...")
        try:
            # Новая версия ChromaDB
            self.chroma_client = chromadb.PersistentClient(path="./claude_rag_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="claude_documents",
                metadata={"description": "Документы для RAG системы Claude"}
            )
            print("Векторная база данных инициализирована")
        except Exception as e:
            print(f"Ошибка при инициализации ChromaDB: {e}")
            # Попробуем альтернативный подход
            try:
                import chromadb.utils.embedding_functions as embedding_functions
                self.chroma_client = chromadb.Client()
                self.collection = self.chroma_client.create_collection(
                    name="claude_documents"
                )
                print("Векторная база данных инициализирована (в памяти)")
            except:
                print("Векторная база недоступна, работаю в упрощенном режиме")
                self.collection = None
        
        # Более подробные документы для демонстрации
        self.sample_documents = [
            """Компания NeuroTech Innovations была основана 15 марта 2015 года в Москве.
            Основатели: Алексей Петров (CEO) и Мария Смирнова (CTO).
            Миссия компании: разработка ИИ решений для медицины.""",
            
            """Основной продукт - платформа NeuroCloud версия 2.1.
            NeuroCloud - это облачная платформа для анализа медицинских изображений и данных пациентов.
            Платформа использует алгоритмы компьютерного зрения и обработки естественного языка.""",
            
            """Технологический стек: Python, PyTorch, TensorFlow, FastAPI, PostgreSQL.
            Архитектура: микросервисы, контейнеризация Docker, оркестрация Kubernetes.
            Модели: ResNet-50, BERT, собственные трансформерные архитектуры.""",
            
            """Финансовые показатели:
            - 2021 год: выручка $10 млн, прибыль $2 млн
            - 2022 год: выручка $15 млн, прибыль $3.5 млн
            - 2023 год: выручка $25 млн, привлечено инвестиций $50 млн
            Инвесторы: Sequoia Capital, Y Combinator, фонд Сбербанка.""",
            
            """Команда: всего 250 сотрудников.
            Распределение: 150 инженеров и исследователей, 50 врачей-консультантов, 
            30 менеджеров продукта, 20 продажников.
            Офисы: Москва (штаб-квартира), Санкт-Петербург, Берлин, Нью-Йорк.""",
            
            """Клиенты и партнеры: 50 больниц в России, 20 клиник в Европе, 
            10 медицинских центров в США. Ключевые партнеры: Mayo Clinic, Charité, 
            московская городская больница №1.""",
            
            """Достижения и награды:
            - 2022: премия "Лучший медицинский стартап" на AI Healthcare Summit
            - 2023: сертификация FDA для диагностики рака легких
            - 2024: точность диагностики достигла 96.5% на тестовых данных
            Опубликовано 15 научных статей в Nature и Science.""",
            
            """Текущие проекты:
            1. NeuroCloud 3.0 - мультимодальный ИИ для диагностики
            2. NeuroGen - генерация персонализированных планов лечения
            3. NeuroScreen - скрининг заболеваний на ранних стадиях
            Ожидаемый запуск: Q4 2024."""
        ]
        
        # Индексируем документы если коллекция пуста
        if self.collection and self.collection.count() == 0:
            self._index_documents(self.sample_documents)
        elif not self.collection:
            print("Режим без векторной базы: RAG будет использовать простой поиск")
            self.documents_index = self._create_simple_index(self.sample_documents)
    
    def _create_simple_index(self, documents: List[str]) -> Dict:
        """Создание простого индекса для работы без ChromaDB"""
        index = {
            "documents": documents,
            "embeddings": []
        }
        
        print("Создаю эмбеддинги для документов...")
        for doc in documents:
            embedding = self.embedding_model.encode(doc).tolist()
            index["embeddings"].append(embedding)
        
        print(f"Создан индекс для {len(documents)} документов")
        return index
    
    def _simple_search(self, query: str, top_k: int = 3) -> List[str]:
        """Простой поиск без ChromaDB"""
        if not hasattr(self, 'documents_index'):
            return []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        embeddings = self.documents_index["embeddings"]
        
        # Вычисляем косинусное сходство
        similarities = []
        for emb in embeddings:
            # Простое вычисление сходства (для демо)
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(similarity)
        
        # Получаем индексы самых релевантных документов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents_index["documents"][i] for i in top_indices]
    
    def _index_documents(self, documents: List[str]):
        """Индексация документов в векторной базе"""
        print("Индексация документов...")
        
        for i, doc in enumerate(documents):
            # Разбиваем документ на смысловые чанки
            sentences = doc.split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Индексируем каждый чанк
            for j, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "doc_id": i,
                        "chunk": j,
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[f"doc_{i}_chunk_{j}"]
                )
        
        print(f"Проиндексировано {len(documents)} документов")
    
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Поиск релевантных чанков по запросу"""
        if self.collection:
            # Используем ChromaDB если доступна
            try:
                query_embedding = self.embedding_model.encode(query).tolist()
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                return results['documents'][0] if results['documents'] else []
            except Exception as e:
                print(f"Ошибка поиска в ChromaDB: {e}")
                return []
        else:
            # Используем простой поиск
            return self._simple_search(query, top_k)
    
    def ask_claude_without_rag(self, question: str) -> str:
        """Запрос к Claude без RAG"""
        try:
            print(f"Запрашиваю Claude (без RAG)...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.7,
                system="Ты - полезный и точный ассистент. Отвечай подробно и информативно.",
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Ошибка при запросе к Claude: {str(e)}"
    
    def ask_claude_with_rag(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Полный RAG-пайплайн с Claude"""
        print(f"🔍 Ищу релевантную информацию в документах...")
        
        # 1. Поиск релевантных чанков
        relevant_chunks = self.search_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            print("Не найдено релевантных документов")
            return self.ask_claude_without_rag(question), []
        
        print(f"Найдено {len(relevant_chunks)} релевантных чанков")
        
        # 2. Формирование контекста
        context = "\n\n".join([f"[Источник {i+1}]:\n{chunk}" 
                              for i, chunk in enumerate(relevant_chunks)])
        
        # 3. Промпт для Claude с контекстом
        prompt = f"""Вот информация из базы знаний компании:

{context}

Вопрос: {question}

Пожалуйста, ответь на вопрос на основе предоставленной информации. 
Если информации недостаточно для полного ответа, скажи об этом.
Будь точен и используй факты из информации."""

        try:
            print(f"Отправляю запрос Claude с контекстом...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                system="Ты отвечаешь строго на основе предоставленного контекста.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text, relevant_chunks
            
        except Exception as e:
            return f"Ошибка при RAG-запросе: {str(e)}", relevant_chunks
    
    def compare_responses(self, question: str):
        """Сравнение ответов с RAG и без RAG"""
        print(f"\n{'='*60}")
        print(f"ВОПРОС: {question}")
        print(f"{'='*60}")
        
        # Ответ без RAG
        print("\n1️⃣  ЗАПРОС БЕЗ RAG (только Claude):")
        answer_without_rag = self.ask_claude_without_rag(question)
        print(f"\n📝 Ответ:\n{'-'*40}")
        print(answer_without_rag)
        
        # Ответ с RAG
        print(f"\n\n2️⃣  ЗАПРОС С RAG (Claude + поиск по документам):")
        answer_with_rag, chunks = self.ask_claude_with_rag(question)
        
        if chunks:
            print(f"\n🔍 Найдено релевантных чанков: {len(chunks)}")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n   Чанк {i} ({len(chunk)} символов):")
                print(f"   {chunk[:150]}..." if len(chunk) > 150 else f"   {chunk}")
        
        print(f"\n📝 Ответ с RAG:\n{'-'*40}")
        print(answer_with_rag)
        
        # Анализ сравнения
        print(f"\n{'='*60}")
        print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print(f"{'='*60}")
        
        self._analyze_and_print(question, answer_without_rag, answer_with_rag)
        
        return {
            "question": question,
            "answer_without_rag": answer_without_rag,
            "answer_with_rag": answer_with_rag,
            "used_chunks": chunks
        }
    
    def _analyze_and_print(self, question: str, answer1: str, answer2: str):
        """Анализ и вывод результатов сравнения"""
        # Простой анализ по длине и содержанию
        len1 = len(answer1)
        len2 = len(answer2)
        
        # Проверяем наличие ключевых фактов из документов
        key_facts = ["2015", "15 марта", "Москв", "NeuroCloud", "2.1", "$50", "250", 
                    "96.5%", "96,5%", "FDA", "Sequoia", "Mayo Clinic", "150 инженер"]
        
        facts_in_rag = sum(1 for fact in key_facts if fact in answer2)
        facts_without = sum(1 for fact in key_facts if fact in answer1)
        
        print(f"\n📈 Сравнение:")
        print(f"   Длина ответа без RAG: {len1} символов")
        print(f"   Длина ответа с RAG: {len2} символов")
        print(f"   Разница: {len2 - len1} символов")
        
        print(f"\n🔍 Ключевые факты:")
        print(f"   Фактов в ответе без RAG: {facts_without}")
        print(f"   Фактов в ответе с RAG: {facts_in_rag}")
        
        # Вывод
        if facts_in_rag > facts_without:
            print(f"\n✅ ВЫВОД: RAG ПОМОГ - добавил конкретные факты и цифры")
        elif len2 > len1 * 1.5:
            print(f"\n⚠️  ВЫВОД: RAG сделал ответ более подробным")
        elif "недостаточно" in answer2.lower() or "нет информации" in answer2.lower():
            print(f"\n❌ ВЫВОД: RAG не нашел нужной информации в документах")
        else:
            print(f"\n➖ ВЫВОД: RAG не дал существенных преимуществ для этого вопроса")
    
    def run_demo_questions(self):
        """Запуск демонстрационных вопросов"""
        demo_questions = [
            "Когда была основана компания NeuroTech Innovations?",
            "Сколько инвестиций привлекла компания в 2023 году?",
            "Сколько сотрудников работает в компании?",
            "Какая точность диагностики у системы?",
            "Какие проекты сейчас в разработке?"
        ]
        
        print("\n" + "="*60)
        print("🧪 ЗАПУСК ДЕМОНСТРАЦИОННЫХ ВОПРОСОВ")
        print("="*60)
        
        results = []
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n\n{'#'*60}")
            print(f"ВОПРОС {i}: {question}")
            print(f"{'#'*60}")
            
            result = self.compare_responses(question)
            results.append(result)
        
        # Краткая статистика
        print("\n" + "="*60)
        print("📊 ИТОГИ ТЕСТИРОВАНИЯ:")
        print("="*60)
        
        helpful = 0
        for result in results:
            rag_answer = result["answer_with_rag"]
            key_facts = ["2015", "$50", "250", "96.5%", "NeuroCloud 3.0"]
            if any(fact in rag_answer for fact in key_facts):
                helpful += 1
        
        print(f"\n✅ RAG помог улучшить ответ в {helpful} из {len(results)} случаев")
        
        if helpful > len(results) / 2:
            print("🎯 RAG система работает эффективно!")
        else:
            print("⚠️  RAG система требует доработки")
    
    def interactive_mode(self):
        """Интерактивный режим"""
        print("\n" + "="*60)
        print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*60)
        print("Выберите действие:")
        print("  1 - Задать вопрос (без RAG)")
        print("  2 - Задать вопрос (с RAG)")
        print("  3 - Сравнить оба подхода")
        print("  4 - Запустить демо-тесты")
        print("  0 - Выход")
        print("="*60)
        
        while True:
            try:
                choice = input("\n👉 Ваш выбор (0-4): ").strip()
                
                if choice == "0":
                    print("\n👋 До свидания!")
                    break
                
                elif choice == "1":
                    question = input("\n🤔 Ваш вопрос: ")
                    answer = self.ask_claude_without_rag(question)
                    print(f"\n📝 ОТВЕТ:\n{'='*40}")
                    print(answer)
                
                elif choice == "2":
                    question = input("\n🤔 Ваш вопрос: ")
                    answer, chunks = self.ask_claude_with_rag(question)
                    if chunks:
                        print(f"\n🔍 Найдено {len(chunks)} релевантных чанков")
                    print(f"\n📝 ОТВЕТ С RAG:\n{'='*40}")
                    print(answer)
                
                elif choice == "3":
                    question = input("\n🤔 Ваш вопрос для сравнения: ")
                    self.compare_responses(question)
                
                elif choice == "4":
                    self.run_demo_questions()
                
                else:
                    print("❌ Неверный выбор. Попробуйте 0-4")
            
            except KeyboardInterrupt:
                print("\n\n👋 Завершение работы...")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")

# Основная часть
if __name__ == "__main__":
    print("🚀 Запуск Claude RAG агента...")
    print("="*60)
    
    try:
        # Инициализация с более простой моделью для начала
        agent = ClaudeRAGAgent(
            api_key=api_key,
            model="claude-3-haiku-20240307"
        )
        
        print("\n" + "="*60)
        print("✅ Агент успешно инициализирован!")
        print(f"🤖 Модель: {agent.model}")
        
        # Автоматически запускаем демо вопросы
        agent.run_demo_questions()
        
        # Затем переходим в интерактивный режим
        agent.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("\n🔧 Устранение неполадок:")
        print("   1. Проверьте API ключ в .env файле")
        print("   2. Проверьте подключение к интернету")
        print("   3. Убедитесь, что установлены все библиотеки")
        print("   4. Попробуйте использовать другую модель Claude")