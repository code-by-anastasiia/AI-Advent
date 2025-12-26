from anthropic import Anthropic
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from typing import List, Dict, Tuple

# ===== ЗАГРУЗКА API КЛЮЧА =====
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("Ошибка: API ключ не найден!")
    exit(1)

class RAGChatBot:
    def __init__(self):
        print("🤖 Инициализация RAG чат-бота...")
        
        # Инициализация Claude
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-3-haiku-20240307"
        
        # Модель для эмбеддингов
        print("🔄 Загрузка модели для поиска...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # База знаний компании
        self.knowledge_base = self._create_knowledge_base()
        
        # Создаем эмбеддинги для базы знаний
        print("📝 Индексация документов...")
        self.knowledge_texts = [doc["content"] for doc in self.knowledge_base]
        self.knowledge_embeddings = self.embedding_model.encode(self.knowledge_texts)
        
        # История диалога
        self.conversation_history = []
        
        # Статистика
        self.stats = {
            "questions_asked": 0,
            "documents_used": 0,
            "sessions": 1
        }
        
        print("✅ RAG чат-бот готов к работе!")
        print(f"📚 База знаний: {len(self.knowledge_base)} документов")
        print("-" * 60)
    
    def _create_knowledge_base(self) -> List[Dict]:
        """Создание базы знаний"""
        return [
            {
                "id": 1,
                "title": "Основание компании",
                "content": "Компания NeuroTech Innovations была основана 15 марта 2015 года в Москве. Основатели: Алексей Петров (CEO) и Мария Смирнова (CTO). Миссия компании - разработка ИИ-решений для медицины.",
                "category": "общая информация",
                "date": "2015-03-15"
            },
            {
                "id": 2,
                "title": "Финансирование и инвестиции",
                "content": "В 2023 году компания привлекла $50 миллионов инвестиций в раунде Series B. Основные инвесторы: Sequoia Capital, фонд Сбербанка и Y Combinator. Общая оценка компании после раунда - $300 миллионов.",
                "category": "финансы",
                "date": "2023-06-20"
            },
            {
                "id": 3,
                "title": "Команда и сотрудники",
                "content": "В компании работает 250 сотрудников. Распределение: 150 инженеров и исследователей, 50 врачей-консультантов, 30 менеджеров продукта, 20 сотрудников отдела продаж. Штаб-квартира находится в Москве, есть офисы в Санкт-Петербурге и Берлине.",
                "category": "команда",
                "date": "2024-01-15"
            },
            {
                "id": 4,
                "title": "Продукт NeuroCloud",
                "content": "Основной продукт - платформа NeuroCloud версии 2.1. Это облачная платформа для анализа медицинских изображений (рентген, МРТ, КТ). Точность диагностики составляет 96.5%. Платформа используется в 50 больницах по России.",
                "category": "продукты",
                "date": "2024-02-01"
            },
            {
                "id": 5,
                "title": "Технологии и исследования",
                "content": "Компания использует модели глубокого обучения на основе трансформеров. Основные технологии: Python, PyTorch, FastAPI, PostgreSQL. Опубликовано 15 научных статей в журналах Nature и Science. Получено 5 патентов на алгоритмы диагностики.",
                "category": "технологии",
                "date": "2024-03-10"
            },
            {
                "id": 6,
                "title": "Партнеры и клиенты",
                "content": "Ключевые партнеры: Mayo Clinic (США), Charité (Германия), Московская городская больница №1. Всего компания сотрудничает с 50 медицинскими учреждениями в России и 20 - за рубежом. В 2024 году планируется выход на рынок Азии.",
                "category": "партнеры",
                "date": "2024-01-30"
            },
            {
                "id": 7,
                "title": "Награды и достижения",
                "content": "2022 - Премия 'Лучший медицинский стартап' на AI Healthcare Summit. 2023 - Сертификация FDA для диагностики рака легких. 2024 - Топ-10 медицинских инноваций по версии Forbes.",
                "category": "достижения",
                "date": "2024-04-05"
            },
            {
                "id": 8,
                "title": "Планы на будущее",
                "content": "В разработке NeuroCloud 3.0 с мультимодальным ИИ. Планируется запуск мобильного приложения для врачей. Цель на 2025 год - охватить 100 больниц в Европе и США.",
                "category": "планы",
                "date": "2024-05-12"
            }
        ]
    
    def _search_in_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict]:
        """Поиск релевантных документов в базе знаний"""
        # Эмбеддинг запроса
        query_embedding = self.embedding_model.encode(query)
        
        # Нормализация
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.knowledge_embeddings / np.linalg.norm(self.knowledge_embeddings, axis=1, keepdims=True)
        
        # Косинусное сходство
        similarities = np.dot(doc_norms, query_norm)
        
        # Получаем топ-K документов
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.knowledge_base[idx].copy()
            doc["similarity"] = float(similarities[idx])
            doc["relevance_percent"] = int(similarities[idx] * 100)
            results.append(doc)
        
        # Фильтруем по порогу релевантности
        threshold = 0.4
        filtered_results = [doc for doc in results if doc["similarity"] >= threshold]
        
        return filtered_results if filtered_results else results[:1]  # Возвращаем хотя бы один
    
    def _format_conversation_history(self, max_messages: int = 6) -> str:
        """Форматирование истории диалога"""
        if not self.conversation_history:
            return ""
        
        # Берем последние N сообщений
        recent_history = self.conversation_history[-max_messages:]
        
        formatted = "ИСТОРИЯ ДИАЛОГА:\n"
        for msg in recent_history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            formatted += f"{role}: {msg['content']}\n"
        
        return formatted
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Форматирование источников для ответа"""
        if not sources:
            return "Источники: информация из общей базы знаний"
        
        formatted = "📚 ИСТОЧНИКИ ИНФОРМАЦИИ:\n"
        for i, source in enumerate(sources, 1):
            formatted += f"\n{i}. 📄 {source['title']}\n"
            formatted += f"   🏷️  Категория: {source['category']}\n"
            formatted += f"   📅 Дата: {source['date']}\n"
            formatted += f"   📊 Релевантность: {source['relevance_percent']}%\n"
            if i < len(sources):
                formatted += "   " + "-" * 40 + "\n"
        
        return formatted
    
    def ask(self, user_message: str) -> Dict:
        """Основной метод для обработки вопроса пользователя"""
        print(f"\n{'='*60}")
        print(f"💬 ВОПРОС: {user_message}")
        print(f"{'='*60}")
        
        # Добавляем вопрос в историю
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Шаг 1: Поиск в базе знаний
        print("🔍 Поиск релевантной информации...")
        relevant_docs = self._search_in_knowledge_base(user_message)
        
        print(f"✅ Найдено релевантных документов: {len(relevant_docs)}")
        for doc in relevant_docs:
            print(f"   • {doc['title']} ({doc['relevance_percent']}% релевантности)")
        
        # Шаг 2: Формируем контекст
        context = "ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ КОМПАНИИ:\n\n"
        for doc in relevant_docs:
            context += f"Документ: {doc['title']}\n"
            context += f"Содержание: {doc['content']}\n\n"
        
        # Шаг 3: Формируем промпт с историей
        history = self._format_conversation_history()
        
        prompt = f"""{history}

{context}

ТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_message}

ИНСТРУКЦИИ:
1. Ответь на вопрос, используя предоставленную информацию из базы знаний
2. Если информации недостаточно, так и скажи
3. Будь точным и конкретным
4. Используй факты и цифры из документов
5. Отвечай на русском языке

ОТВЕТ:"""
        
        # Шаг 4: Запрос к Claude
        print("🤖 Генерация ответа...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
                system="Ты - ассистент компании NeuroTech, который отвечает на вопросы сотрудников и клиентов на основе базы знаний компании.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            answer = response.content[0].text
            
        except Exception as e:
            answer = f"Извините, произошла ошибка при обработке запроса: {str(e)}"
        
        # Шаг 5: Форматируем финальный ответ с источниками
        sources_text = self._format_sources(relevant_docs)
        full_response = f"{answer}\n\n{sources_text}"
        
        # Добавляем ответ в историю
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "sources": [doc["id"] for doc in relevant_docs]
        })
        
        # Обновляем статистику
        self.stats["questions_asked"] += 1
        self.stats["documents_used"] += len(relevant_docs)
        
        return {
            "answer": answer,
            "sources": relevant_docs,
            "sources_text": sources_text,
            "full_response": full_response,
            "stats": self.stats.copy()
        }
    
    def clear_history(self):
        """Очистка истории диалога"""
        self.conversation_history = []
        print("🗑️  История диалога очищена")
    
    def show_stats(self):
        """Показать статистику"""
        print("\n📊 СТАТИСТИКА ЧАТ-БОТА:")
        print(f"   • Задано вопросов: {self.stats['questions_asked']}")
        print(f"   • Использовано документов: {self.stats['documents_used']}")
        print(f"   • Сессий: {self.stats['sessions']}")
        print(f"   • Сообщений в истории: {len(self.conversation_history)}")
        
        if self.conversation_history:
            last_time = self.conversation_history[-1]['timestamp']
            print(f"   • Последнее сообщение: {last_time[:19]}")
    
    def save_conversation(self, filename: str = None):
        """Сохранение диалога в файл"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        data = {
            "conversation": self.conversation_history,
            "stats": self.stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Диалог сохранен в файл: {filename}")
        return filename
    
    def show_knowledge_base(self):
        """Показать базу знаний"""
        print("\n📚 БАЗА ЗНАНИЙ КОМПАНИИ:")
        print(f"Всего документов: {len(self.knowledge_base)}")
        print("-" * 60)
        
        for doc in self.knowledge_base:
            print(f"\n📄 Документ #{doc['id']}: {doc['title']}")
            print(f"   🏷️  Категория: {doc['category']}")
            print(f"   📅 Дата: {doc['date']}")
            print(f"   📝 {doc['content'][:150]}...")

def main():
    """Главная функция - интерактивный чат"""
    print("💬 НЕЙРОТЕХ ЧАТ-БОТ С ПАМЯТЬЮ И RAG")
    print("="*60)
    print("Я - ассистент компании NeuroTech Innovations.")
    print("Отвечаю на вопросы о компании, используя базу знаний.")
    print("="*60)
    
    # Создаем бота
    bot = RAGChatBot()
    
    # Демонстрационные вопросы для примера
    demo_questions = [
        "Когда была основана компания?",
        "Сколько инвестиций вы привлекли?",
        "Сколько сотрудников у вас работает?",
        "Что такое NeuroCloud?",
        "С какими больницами вы сотрудничаете?"
    ]
    
    print("\n💡 Примеры вопросов, которые можно задать:")
    for i, question in enumerate(demo_questions, 1):
        print(f"   {i}. {question}")
    
    print("\n" + "="*60)
    print("🎮 КОМАНДЫ ЧАТА:")
    print("   /help     - Показать помощь")
    print("   /stats    - Показать статистику")
    print("   /clear    - Очистить историю")
    print("   /save     - Сохранить диалог")
    print("   /kb       - Показать базу знаний")
    print("   /demo     - Запустить демо-диалог")
    print("   /exit     - Выйти из чата")
    print("="*60)
    
    # Главный цикл чата
    while True:
        try:
            # Ввод пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            if not user_input:
                continue
            
            # Обработка команд
            if user_input.lower() == '/exit':
                print("\n👋 До свидания! Спасибо за общение!")
                bot.save_conversation()
                break
            
            elif user_input.lower() == '/help':
                print("\n📋 ПОМОЩЬ:")
                print("   • Задавайте вопросы о компании NeuroTech")
                print("   • Бот ищет ответы в базе знаний")
                print("   • Каждый ответ содержит источники информации")
                print("   • Бот помнит историю разговора")
                print("\n💡 Примеры вопросов:")
                for q in demo_questions:
                    print(f"   - {q}")
                continue
            
            elif user_input.lower() == '/stats':
                bot.show_stats()
                continue
            
            elif user_input.lower() == '/clear':
                bot.clear_history()
                continue
            
            elif user_input.lower() == '/save':
                filename = bot.save_conversation()
                print(f"✅ Диалог сохранен как {filename}")
                continue
            
            elif user_input.lower() == '/kb':
                bot.show_knowledge_base()
                continue
            
            elif user_input.lower() == '/demo':
                print("\n🧪 ЗАПУСК ДЕМО-ДИАЛОГА...")
                for question in demo_questions:
                    print(f"\n{'='*60}")
                    print(f"👤 Вы: {question}")
                    
                    response = bot.ask(question)
                    print(f"\n🤖 Бот: {response['answer'][:200]}...")
                    
                    # Показываем источники кратко
                    if response['sources']:
                        print(f"\n📚 Использовано источников: {len(response['sources'])}")
                        for source in response['sources']:
                            print(f"   • {source['title']} ({source['relevance_percent']}%)")
                
                print("\n✅ Демо-диалог завершен")
                continue
            
            # Обычный вопрос пользователя
            print(f"\n{'='*60}")
            print(f"👤 ВАШ ВОПРОС: {user_input}")
            print(f"{'='*60}")
            
            # Получаем ответ от бота
            response = bot.ask(user_input)
            
            # Выводим ответ
            print(f"\n{'='*60}")
            print("🤖 ОТВЕТ БОТА:")
            print(f"{'='*60}")
            print(response['answer'])
            
            # Выводим источники
            print(f"\n{'='*60}")
            print("📚 ИСТОЧНИКИ ИНФОРМАЦИИ:")
            print(f"{'='*60}")
            print(response['sources_text'])
            
            # Краткая статистика
            print(f"\nℹ️  Для этого ответа использовано {len(response['sources'])} документов")
            print(f"📈 Всего вопросов в диалоге: {response['stats']['questions_asked']}")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Прервано пользователем")
            save = input("Сохранить диалог перед выходом? (да/нет): ").lower()
            if save in ['да', 'д', 'yes', 'y']:
                bot.save_conversation()
            print("👋 До свидания!")
            break
        
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("Попробуйте еще раз или введите /help для помощи")

if __name__ == "__main__":
    main()