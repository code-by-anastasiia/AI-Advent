import json
import os
import requests
from pathlib import Path

class ProjectRAG:
    """
    Простая RAG система для документации проекта
    """
    
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.documents = []
        
    def index_project(self):
        """
        Индексируем документацию: README и docstrings из .py файлов
        """
        print(f"Индексирую проект: {self.project_path}")
        
        # 1. Индексируем README
        readme_path = self.project_path / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding='utf-8')
            self.documents.append({
                "source": "README.md",
                "content": content,
                "embedding": self._get_embedding(content)
            })
            print(f"✓ Проиндексирован README.md")
        
        # 2. Индексируем Python файлы
        py_files = list(self.project_path.rglob("*.py"))
        for py_file in py_files:
            # Пропускаем служебные папки
            if any(skip in py_file.parts for skip in ["venv", ".git", "__pycache__"]):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                # Берем только если есть docstring или комментарии
                if '"""' in content or "'''" in content or "#" in content:
                    self.documents.append({
                        "source": str(py_file.relative_to(self.project_path)),
                        "content": content[:2000],  # Первые 2000 символов
                        "embedding": self._get_embedding(content[:2000])
                    })
                    print(f"✓ Проиндексирован {py_file.name}")
            except Exception as e:
                print(f"✗ Ошибка при индексации {py_file.name}: {e}")
        
        print(f"Всего проиндексировано документов: {len(self.documents)}")
        
    def _get_embedding(self, text):
        """
        Получаем embedding через Ollama
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            else:
                return [0.0] * 768  # Заглушка
        except Exception as e:
            print(f"Ошибка получения embedding: {e}")
            return [0.0] * 768
    
    def _cosine_similarity(self, vec1, vec2):
        """
        Косинусное сходство
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query, top_k=3):
        """
        Ищем релевантные документы для запроса
        """
        if not self.documents:
            return "Документация не проиндексирована"
        
        query_emb = self._get_embedding(query)
        
        # Считаем similarity для каждого документа
        results = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_emb, doc["embedding"])
            if similarity > 0.2:  # Порог релевантности
                results.append({
                    "source": doc["source"],
                    "content": doc["content"],
                    "similarity": similarity
                })
        
        # Сортируем и берем топ
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]
        
        if not top_results:
            return "Релевантной документации не найдено"
        
        # Форматируем для промпта
        formatted = "\n\n---\n\n".join([
            f"Источник: {r['source']}\n\n{r['content']}"
            for r in top_results
        ])
        
        return formatted
    