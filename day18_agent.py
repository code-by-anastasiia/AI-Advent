from anthropic import Anthropic
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== ЗАГРУЗКА API КЛЮЧА =====
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Ошибка: API ключ не найден!")
    exit(1)

# Простой RAG с фильтрацией
class SimpleRAG:
    def __init__(self):
        print("🚀 Инициализация RAG системы...")
        
        self.client = Anthropic(api_key=api_key)
        
        # Загружаем модель для эмбеддингов
        print("🔄 Загрузка модели для эмбеддингов...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Наша база знаний
        self.documents = [
            "Компания NeuroTech Innovations была основана 15 марта 2015 года в Москве.",
            "Основатели компании: Алексей Петров (CEO) и Мария Смирнова (CTO).",
            "В 2023 году компания привлекла $50 миллионов инвестиций.",
            "Основные инвесторы: Sequoia Capital и фонд Сбербанка.",
            "В компании работает 250 сотрудников.",
            "Из них 150 - инженеры и исследователи ИИ.",
            "Основной продукт - платформа NeuroCloud версии 2.1.",
            "Платформа анализирует медицинские изображения с точностью 96.5%.",
            "Искусственный интеллект - это область компьютерных наук.",
            "Машинное обучение является подразделом искусственного интеллекта.",
            "Стартапы часто привлекают венчурные инвестиции.",
            "Кремниевая долина - центр технологических инноваций."
        ]
        
        # Создаем эмбеддинги для всех документов
        print("📝 Создание эмбеддингов для документов...")
        self.document_embeddings = self.embedding_model.encode(self.documents)
        
        print(f"✅ Система готова! Загружено {len(self.documents)} документов")
        print("-" * 50)
    
    def calculate_similarity(self, query, document_embeddings):
        """Вычисление косинусного сходства"""
        query_embedding = self.embedding_model.encode(query)
        
        # Нормализация векторов
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        
        # Косинусное сходство
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def search_without_filter(self, query, top_k=5):
        """Поиск без фильтрации"""
        print(f"\n🔍 ПОИСК БЕЗ ФИЛЬТРАЦИИ")
        print(f"Запрос: '{query}'")
        
        similarities = self.calculate_similarity(query, self.document_embeddings)
        
        # Получаем топ-K документов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print(f"Найдено документов: {len(top_indices)}")
        print("Топ документы:")
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            doc = self.documents[idx]
            results.append((doc, similarity))
            
            print(f"{i}. [Сходство: {similarity:.3f}] {doc}")
        
        return results
    
    def search_with_filter(self, query, threshold=0.5, top_k=10):
        """Поиск с фильтрацией по порогу"""
        print(f"\n🔍 ПОИСК С ФИЛЬТРАЦИЕЙ (порог: {threshold})")
        print(f"Запрос: '{query}'")
        
        similarities = self.calculate_similarity(query, self.document_embeddings)
        
        # Фильтрация по порогу
        filtered_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
        
        if not filtered_indices:
            print(f"❌ Нет документов с сходством >= {threshold}")
            return []
        
        # Сортируем отфильтрованные документы
        filtered_similarities = [similarities[i] for i in filtered_indices]
        sorted_indices = [x for _, x in sorted(zip(filtered_similarities, filtered_indices), reverse=True)]
        
        # Берем топ-K
        top_indices = sorted_indices[:top_k]
        
        print(f"Всего документов: {len(self.documents)}")
        print(f"После фильтрации: {len(top_indices)}/{len(filtered_indices)}")
        print("Отфильтрованные документы:")
        
        results = []
        for i, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            doc = self.documents[idx]
            results.append((doc, similarity))
            
            print(f"{i}. [Сходство: {similarity:.3f}] {doc}")
        
        return results
    
    def ask_claude(self, query, context=""):
        """Запрос к Claude"""
        try:
            if context:
                prompt = f"""Используй следующую информацию для ответа на вопрос:
                
{context}

Вопрос: {query}

Ответь на основе предоставленной информации. Если информации недостаточно, скажи об этом."""
            else:
                prompt = query
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def compare_approaches(self, query):
        """Сравнение подходов с фильтрацией и без"""
        print("\n" + "="*60)
        print("🔄 СРАВНЕНИЕ ПОДХОДОВ")
        print("="*60)
        
        # 1. Без фильтрации
        print("\n1️⃣  БЕЗ ФИЛЬТРАЦИИ:")
        results_no_filter = self.search_without_filter(query)
        
        if results_no_filter:
            context_no_filter = "\n".join([doc for doc, _ in results_no_filter[:3]])
            answer_no_filter = self.ask_claude(query, context_no_filter)
            print(f"\n📝 ОТВЕТ БЕЗ ФИЛЬТРАЦИИ:")
            print(answer_no_filter)
        
        # 2. С фильтрацией
        print("\n2️⃣  С ФИЛЬТРАЦИЕЙ (порог 0.5):")
        results_with_filter = self.search_with_filter(query, threshold=0.5)
        
        if results_with_filter:
            context_with_filter = "\n".join([doc for doc, _ in results_with_filter[:3]])
            answer_with_filter = self.ask_claude(query, context_with_filter)
            print(f"\n📝 ОТВЕТ С ФИЛЬТРАЦИЕЙ:")
            print(answer_with_filter)
        
        # 3. Анализ
        print("\n" + "="*60)
        print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        if results_no_filter and results_with_filter:
            print(f"• Документов без фильтра: {len(results_no_filter)}")
            print(f"• Документов с фильтром: {len(results_with_filter)}")
            print(f"• Удалено документов: {len(results_no_filter) - len(results_with_filter)}")
            
            # Проверяем качество
            key_terms = ["2015", "15 марта", "$50", "250", "NeuroCloud", "96.5%"]
            has_key_terms_no_filter = any(term in answer_no_filter for term in key_terms)
            has_key_terms_with_filter = any(term in answer_with_filter for term in key_terms)
            
            print(f"\n🔑 Ключевые факты в ответах:")
            print(f"   Без фильтра: {'✅' if has_key_terms_no_filter else '❌'}")
            print(f"   С фильтром: {'✅' if has_key_terms_with_filter else '❌'}")
            
            if has_key_terms_with_filter and not has_key_terms_no_filter:
                print("\n🎯 ВЫВОД: ФИЛЬТРАЦИЯ ПОМОГЛА - ответ стал более точным!")
            elif has_key_terms_with_filter and has_key_terms_no_filter:
                print("\n⚖️  ВЫВОД: Оба подхода дали хорошие результаты")
            else:
                print("\n⚠️  ВЫВОД: Фильтрация не улучшила результат")
        
        elif results_with_filter and not results_no_filter:
            print("❌ Без фильтра не найдено документов, но с фильтром - найдены!")
        elif not results_with_filter and results_no_filter:
            print("⚠️  Фильтрация удалила ВСЕ документы - возможно, порог слишком высокий")
        else:
            print("❌ Не найдено документов ни с одним из подходов")

# Главная функция
def main():
    print("🤖 RAG СИСТЕМА С ФИЛЬТРАЦИЕЙ И РЕРАНКИНГОМ")
    print("="*60)
    
    # Создаем RAG систему
    rag = SimpleRAG()
    
    # Тестовые вопросы
    test_questions = [
        "Когда была основана компания NeuroTech?",
        "Сколько инвестиций привлекла компания?",
        "Сколько сотрудников работает в компании?",
        "Какой основной продукт у компании и какая у него точность?",
        "Что такое искусственный интеллект?"  # Общий вопрос
    ]
    
    # Автоматический тест
    print("\n🧪 ЗАПУСК АВТОМАТИЧЕСКОГО ТЕСТА")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'#'*60}")
        print(f"ТЕСТ {i}/{len(test_questions)}")
        print(f"ВОПРОС: {question}")
        print(f"{'#'*60}")
        
        rag.compare_approaches(question)
        
        # Пауза между вопросами
        if i < len(test_questions):
            input("\nНажми Enter для продолжения...")
    
    # Интерактивный режим
    print("\n" + "="*60)
    print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    
    while True:
        print("\nВыберите действие:")
        print("1 - Задать новый вопрос")
        print("2 - Изменить порог фильтрации")
        print("3 - Показать все документы")
        print("0 - Выход")
        
        choice = input("\n👉 Ваш выбор: ").strip()
        
        if choice == "1":
            question = input("\n🤔 Введите ваш вопрос: ")
            rag.compare_approaches(question)
        
        elif choice == "2":
            try:
                new_threshold = float(input(f"\n📏 Введите новый порог (текущий 0.5, от 0 до 1): "))
                if 0 <= new_threshold <= 1:
                    print(f"\n🔄 Тестируем с порогом {new_threshold}...")
                    question = "Когда была основана компания NeuroTech?"
                    rag.search_with_filter(question, threshold=new_threshold)
                else:
                    print("❌ Порог должен быть между 0 и 1")
            except:
                print("❌ Введите число")
        
        elif choice == "3":
            print(f"\n📚 ВСЕ ДОКУМЕНТЫ ({len(rag.documents)}):")
            for i, doc in enumerate(rag.documents, 1):
                print(f"{i}. {doc}")
        
        elif choice == "0":
            print("\n👋 Завершение работы...")
            break
        
        else:
            print("❌ Неверный выбор")

if __name__ == "__main__":
    main()