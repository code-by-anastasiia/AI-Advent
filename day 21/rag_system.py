import sys
from pathlib import Path
import ollama

class CodeStandardsRAG:
    """RAG система для поиска релевантных стандартов кодирования"""
    
    def __init__(self, standards_folder: str = "code_standards"):
        self.standards_folder = Path(standards_folder)
        self.chunks = {}  # {id: текст}
        self.embeddings = {}  # {id: вектор}
        self.metadata = {}  # {id: {'file': ..., 'section': ...}}
        
        print(f"[RAG] Initializing from {standards_folder}", file=sys.stderr)
        self._load_standards()
        print(f"[RAG] Loaded {len(self.chunks)} chunks", file=sys.stderr)
    
    def _load_standards(self):
        """Загружает все markdown файлы со стандартами"""
        
        if not self.standards_folder.exists():
            print(f"[RAG] WARNING: Folder {self.standards_folder} not found!", file=sys.stderr)
            return
        
        chunk_id = 0
        
        for md_file in self.standards_folder.glob("*.md"):
            print(f"[RAG] Processing {md_file.name}...", file=sys.stderr)
            
            content = md_file.read_text(encoding='utf-8')
            sections = self._split_by_headers(content)
            
            for section_title, section_text in sections:
                if len(section_text.strip()) < 20:
                    continue  # Пропускаем слишком короткие секции
                
                # Разбиваем длинные секции на чанки
                text_chunks = self._split_into_chunks(section_text)
                
                for chunk_text in text_chunks:
                    # Сохраняем чанк
                    self.chunks[chunk_id] = chunk_text
                    self.metadata[chunk_id] = {
                        'file': md_file.name,
                        'section': section_title
                    }
                    
                    # Создаём embedding
                    self._create_embedding(chunk_id, chunk_text)
                    
                    chunk_id += 1
    
    def _split_by_headers(self, text: str) -> list[tuple[str, str]]:
        """Разбивает markdown по заголовкам"""
        lines = text.split('\n')
        sections = []
        current_title = "Introduction"
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Сохраняем предыдущую секцию
                if current_content:
                    sections.append((
                        current_title,
                        '\n'.join(current_content)
                    ))
                
                # Новая секция
                current_title = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Последняя секция
        if current_content:
            sections.append((
                current_title,
                '\n'.join(current_content)
            ))
        
        return sections
    
    def _split_into_chunks(self, text: str, max_chars: int = 500) -> list[str]:
        """Разбивает текст на чанки по предложениям"""
        
        # Разбиваем на параграфы
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if current_length + para_length > max_chars and current_chunk:
                # Сохраняем текущий чанк
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Последний чанк
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    def _create_embedding(self, chunk_id: int, text: str):
        """Создаёт embedding для чанка"""
        try:
            response = ollama.embeddings(
                model='nomic-embed-text',
                prompt=text
            )
            self.embeddings[chunk_id] = response['embedding']
            
        except Exception as e:
            print(f"[RAG] Embedding error for chunk {chunk_id}: {e}", file=sys.stderr)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Ищет наиболее релевантные стандарты
        
        Args:
            query: Поисковый запрос (обычно код или описание проблемы)
            top_k: Количество результатов
        
        Returns:
            Список словарей с ключами: text, metadata, score
        """
        if not self.embeddings:
            print("[RAG] No embeddings available!", file=sys.stderr)
            return []
        
        print(f"[RAG] Searching for: {query[:50]}...", file=sys.stderr)
        
        # Создаём embedding запроса
        try:
            query_response = ollama.embeddings(
                model='nomic-embed-text',
                prompt=query
            )
            query_embedding = query_response['embedding']
        except Exception as e:
            print(f"[RAG] Query embedding error: {e}", file=sys.stderr)
            return []
        
        # Считаем similarity со всеми чанками
        similarities = {}
        for chunk_id, embedding in self.embeddings.items():
            sim = self._cosine_similarity(query_embedding, embedding)
            similarities[chunk_id] = sim
        
        # Сортируем и берём top-k
        top_ids = sorted(
            similarities.keys(),
            key=lambda x: similarities[x],
            reverse=True
        )[:top_k]
        
        results = []
        for chunk_id in top_ids:
            results.append({
                'text': self.chunks[chunk_id],
                'metadata': self.metadata[chunk_id],
                'score': similarities[chunk_id]
            })
        
        print(f"[RAG] Found {len(results)} relevant chunks", file=sys.stderr)
        
        return results
    
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Вычисляет косинусное сходство между векторами"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    