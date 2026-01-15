#!/usr/bin/env python3
"""
RAG система для поиска в документации проекта AI агентов.
Использует Ollama для создания embeddings и cosine similarity для поиска.
"""

import json
import sys
from pathlib import Path
import requests
import numpy as np

class ProjectRAG:
    def __init__(self, knowledge_base_dir: str, model: str = "nomic-embed-text"):
        """
        Инициализация RAG системы
        
        Args:
            knowledge_base_dir: Путь к папке с документацией
            model: Модель Ollama для embeddings
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.model = model
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
        # Хранилище документов и их embeddings
        self.documents = []
        self.embeddings = []
        
        print("[RAG] Инициализация...", file=sys.stderr)
    
    def _get_embedding(self, text: str) -> list:
        """Получить embedding для текста через Ollama"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model, "prompt": text},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                print(f"[RAG ERROR] Ollama ответил {response.status_code}", file=sys.stderr)
                return None
                
        except Exception as e:
            print(f"[RAG ERROR] {e}", file=sys.stderr)
            return None
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Вычислить cosine similarity между двумя векторами"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def index_documents(self):
        """Индексировать все документы из knowledge_base_dir"""
        print(f"[RAG] Индексация документов из {self.knowledge_base_dir}", file=sys.stderr)
        
        # Ищем все .md файлы
        md_files = list(self.knowledge_base_dir.glob("*.md"))
        
        if not md_files:
            print("[RAG WARNING] Документы не найдены!", file=sys.stderr)
            return
        
        for file_path in md_files:
            try:
                # Читаем файл
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Создаем embedding
                print(f"[RAG] Обрабатываю {file_path.name}...", file=sys.stderr)
                embedding = self._get_embedding(content)
                
                if embedding:
                    self.documents.append({
                        "filename": file_path.name,
                        "content": content,
                        "path": str(file_path)
                    })
                    self.embeddings.append(embedding)
                    print(f"[RAG] ✓ {file_path.name} добавлен", file=sys.stderr)
                else:
                    print(f"[RAG] ✗ Не удалось создать embedding для {file_path.name}", file=sys.stderr)
                    
            except Exception as e:
                print(f"[RAG ERROR] Ошибка при обработке {file_path.name}: {e}", file=sys.stderr)
        
        print(f"[RAG] Индексация завершена: {len(self.documents)} документов", file=sys.stderr)
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.3):
        """
        Поиск релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            threshold: Минимальная релевантность (0-1)
            
        Returns:
            Список словарей с filename, content, score
        """
        if not self.documents:
            print("[RAG] База документов пуста!", file=sys.stderr)
            return []
        
        # Получаем embedding для запроса
        query_embedding = self._get_embedding(query)
        
        if not query_embedding:
            print("[RAG] Не удалось создать embedding для запроса", file=sys.stderr)
            return []
        
        # Вычисляем similarity для всех документов
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append({
                "index": i,
                "score": score
            })
        
        # Сортируем по релевантности
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Фильтруем по threshold и берем top_k
        results = []
        for sim in similarities[:top_k]:
            if sim["score"] >= threshold:
                doc = self.documents[sim["index"]]
                results.append({
                    "filename": doc["filename"],
                    "content": doc["content"],
                    "score": round(sim["score"], 3)
                })
                print(f"[RAG] Найден: {doc['filename']} (score: {sim['score']:.3f})", file=sys.stderr)
        
        if not results:
            print(f"[RAG] Релевантные документы не найдены (все < {threshold})", file=sys.stderr)
        
        return results

# Тестирование (если запускаем напрямую)
if __name__ == "__main__":
    # Создаем RAG
    rag = ProjectRAG("knowledge_base")
    
    # Индексируем документы
    rag.index_documents()
    
    # Тестовый поиск
    print("\n=== Тест поиска ===\n")
    
    test_queries = [
        "Что такое RAG?",
        "Как создать задачу?",
        "Какие проекты уже завершены?"
    ]
    
    for query in test_queries:
        print(f"\nЗапрос: {query}")
        results = rag.search(query, top_k=2)
        
        if results:
            for r in results:
                print(f"  - {r['filename']} (релевантность: {r['score']})")
        else:
            print("  Ничего не найдено")