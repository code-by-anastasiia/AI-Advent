"""
ИНДЕКСАЦИЯ РЕАЛЬНЫХ ДОКУМЕНТОВ
Читает файлы из папки 'documents' и индексирует их
"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
from pathlib import Path

def split_into_chunks(text, chunk_size=300, overlap=50):
    """Разбивает текст на чанки с перекрытием"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Пытаемся разбить по предложению
        if end < len(text):
            last_period = max(chunk.rfind('. '), chunk.rfind('! '), chunk.rfind('? '))
            if last_period > chunk_size * 0.5:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def load_documents_from_folder(folder_path):
    """Загружает все текстовые файлы из папки"""
    documents = []
    
    print(f"\n📂 Сканируем папку: {folder_path}")
    
    # Поддерживаемые форматы
    text_extensions = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.rst']
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Папка не найдена: {folder_path}")
        return documents
    
    # Ищем все текстовые файлы
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            print(f"   📄 Найден: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'content': content
                })
                
            except Exception as e:
                print(f"   ⚠️ Ошибка чтения {file_path.name}: {e}")
    
    return documents


def main():
    print("=" * 70)
    print("  ИНДЕКСАЦИЯ ДОКУМЕНТОВ - Работа с файлами")
    print("=" * 70)
    
    # НАСТРОЙКИ - измените под себя
    DOCUMENTS_FOLDER = "documents"  # Папка с вашими документами
    INDEX_FILE = "document_index.json"  # Куда сохранить индекс
    CHUNK_SIZE = 300  # Размер чанка в символах
    
    # Создаём папку для документов, если её нет
    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"\n📁 Создаём папку: {DOCUMENTS_FOLDER}")
        os.makedirs(DOCUMENTS_FOLDER)
        
        # Создаём пример файлов для демонстрации
        print("📝 Создаём примеры файлов...")
        
        with open(f"{DOCUMENTS_FOLDER}/README.md", "w", encoding="utf-8") as f:
            f.write("""# Мой AI Проект

Это проект по созданию RAG системы для работы с документами.

## Возможности

- Индексация текстовых документов
- Семантический поиск
- Генерация ответов на основе контекста

## Технологии

- Python 3.8+
- Sentence Transformers
- FAISS для векторного поиска
""")
        
        with open(f"{DOCUMENTS_FOLDER}/article.txt", "w", encoding="utf-8") as f:
            f.write("""Машинное обучение в современном мире

Машинное обучение становится всё более важной технологией. 
Оно позволяет компьютерам учиться на данных без явного программирования.

Основные направления:
- Supervised Learning (обучение с учителем)
- Unsupervised Learning (обучение без учителя)  
- Reinforcement Learning (обучение с подкреплением)

Применение машинного обучения можно найти везде: от рекомендательных 
систем до автономных автомобилей.
""")
        
        with open(f"{DOCUMENTS_FOLDER}/code_example.py", "w", encoding="utf-8") as f:
            f.write("""# Пример кода для загрузки данных

import pandas as pd

def load_data(file_path):
    \"\"\"Загружает данные из CSV файла\"\"\"
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    \"\"\"Предобработка данных\"\"\"
    # Удаляем пропущенные значения
    df = df.dropna()
    # Нормализуем данные
    df = (df - df.mean()) / df.std()
    return df
""")
        
        print(f"✅ Созданы примеры файлов в папке '{DOCUMENTS_FOLDER}'")
        print(f"   Вы можете заменить их своими файлами!")
    
    # ШАГ 1: Загрузка документов
    documents = load_documents_from_folder(DOCUMENTS_FOLDER)
    
    if not documents:
        print("\n❌ Не найдено документов для индексации!")
        print(f"   Добавьте .txt, .md, .py или другие файлы в папку '{DOCUMENTS_FOLDER}'")
        return
    
    print(f"\n✅ Загружено {len(documents)} документов")
    
    # ШАГ 2: Загрузка модели
    print("\n📥 Загружаем модель для создания эмбеддингов...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Модель загружена!")
    
    # ШАГ 3: Разбивка на чанки
    print(f"\n✂️  Разбиваем документы на чанки (размер: {CHUNK_SIZE} символов)...")
    
    all_chunks = []
    
    for doc_id, doc in enumerate(documents):
        chunks = split_into_chunks(doc['content'], chunk_size=CHUNK_SIZE)
        print(f"   {doc['filename']}: {len(chunks)} чанков")
        
        for chunk_id, chunk_text in enumerate(chunks):
            all_chunks.append({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'filename': doc['filename'],
                'text': chunk_text
            })
    
    print(f"\n✅ Всего создано {len(all_chunks)} чанков")
    
    # ШАГ 4: Создание эмбеддингов
    print("\n🔢 Создаём эмбеддинги для всех чанков...")
    
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    
    print(f"✅ Создано {len(embeddings)} эмбеддингов")
    print(f"   Размерность каждого: {embeddings.shape[1]} чисел")
    
    # ШАГ 5: Сохранение индекса
    print(f"\n💾 Сохраняем индекс в файл: {INDEX_FILE}")
    
    index_data = {
        'documents': [
            {'filename': doc['filename'], 'path': doc['path']}
            for doc in documents
        ],
        'chunks': all_chunks,
        'embeddings': embeddings.tolist(),
        'config': {
            'chunk_size': CHUNK_SIZE,
            'model': "paraphrase-multilingual-MiniLM-L12-v2",
            'embedding_dim': embeddings.shape[1]
        }
    }
    
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(INDEX_FILE) / 1024  # KB
    print(f"✅ Индекс сохранён!")
    print(f"   Размер файла: {file_size:.2f} KB")
    
    # ШАГ 6: Демонстрация поиска
    print("\n" + "=" * 70)
    print("  ТЕСТИРОВАНИЕ ПОИСКА")
    print("=" * 70)
    
    test_queries = [
        "Что такое машинное обучение?",
        "Как загрузить данные?",
        "Возможности проекта"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Запрос: '{query}'")
        
        # Создаём эмбеддинг для запроса
        query_embedding = model.encode([query])[0]
        
        # Считаем похожесть со всеми чанками
        similarities = []
        for emb in embeddings:
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(similarity)
        
        # Находим топ-2 самых похожих
        top_indices = np.argsort(similarities)[-2:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            chunk = all_chunks[idx]
            similarity = similarities[idx]
            
            print(f"\n   {rank}. Файл: {chunk['filename']}")
            print(f"      Похожесть: {similarity:.3f}")
            print(f"      Текст: {chunk['text'][:120]}...")
    
    # Финальная сводка
    print("\n" + "=" * 70)
    print("  ✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\n📊 Статистика:")
    print(f"   • Документов: {len(documents)}")
    print(f"   • Чанков: {len(all_chunks)}")
    print(f"   • Эмбеддингов: {len(embeddings)}")
    print(f"   • Индекс сохранён: {INDEX_FILE}")
    print(f"\n📁 Ваши документы в папке: {DOCUMENTS_FOLDER}")
    print(f"💾 Индекс готов к использованию в RAG системе!")


if __name__ == "__main__":
    main()
