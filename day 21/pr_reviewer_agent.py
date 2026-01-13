import anthropic
import os
import sys
from pathlib import Path
from rag_system import CodeStandardsRAG
from dotenv import load_dotenv

load_dotenv()

class PRReviewerAgent:
    """Агент для автоматического ревью Pull Request"""
    
    def __init__(self):
        print("\n" + "="*70, file=sys.stderr)
        print("PR REVIEWER AGENT INITIALIZATION", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment!")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        print("[Agent] Initializing RAG system...", file=sys.stderr)
        self.rag = CodeStandardsRAG("code_standards")
        
        print("[Agent] Initialization complete!\n", file=sys.stderr)
    
    def review_pr(self) -> str:
        """Выполняет полное ревью локального PR"""
        print("="*70, file=sys.stderr)
        print("STARTING PR REVIEW", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr)
        
        print("[Step 1] Reading PR data from files...", file=sys.stderr)
        pr_data = self._read_pr_data()
        
        print("\n[Step 2] Finding relevant code standards via RAG...", file=sys.stderr)
        standards = self._find_relevant_standards(pr_data)
        
        print("\n[Step 3] Generating code review...", file=sys.stderr)
        review = self._generate_review(pr_data, standards)
        
        output_file = "pr_review_result.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(review)
        
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"✅ Review completed! Saved to {output_file}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        
        return review
    
    def _read_pr_data(self) -> str:
        """Читает данные PR из локальных файлов"""
        
        # Читаем описание PR
        desc_path = Path("mock_pr/pr_description.txt")
        description = desc_path.read_text(encoding='utf-8')
        
        # Читаем старый код
        old_path = Path("mock_pr/old_version.py")
        old_code = old_path.read_text(encoding='utf-8')
        
        # Читаем новый код
        new_path = Path("mock_pr/new_version.py")
        new_code = new_path.read_text(encoding='utf-8')
        
        # Формируем единый текст
        pr_data = f"""# Pull Request Information

{description}

---

## Old Version (Before PR)
```python
{old_code}
```

---

## New Version (After PR)
```python
{new_code}
```
"""
        
        print(f"  ✓ PR data loaded ({len(pr_data)} chars)", file=sys.stderr)
        return pr_data
    
    def _find_relevant_standards(self, pr_data: str) -> str:
        """Использует RAG для поиска релевантных стандартов"""
        
        results = self.rag.search(pr_data, top_k=5)
        
        if not results:
            print("  ⚠ No relevant standards found", file=sys.stderr)
            return "No specific standards found."
        
        standards_text = "# Relevant Code Standards\n\n"
        
        for i, result in enumerate(results, 1):
            standards_text += f"## Standard {i}\n"
            standards_text += f"**Source:** {result['metadata']['file']}\n"
            standards_text += f"**Section:** {result['metadata']['section']}\n"
            standards_text += f"**Relevance:** {result['score']:.3f}\n\n"
            standards_text += result['text']
            standards_text += "\n\n" + "---" + "\n\n"
        
        print(f"  ✓ Found {len(results)} relevant standards", file=sys.stderr)
        
        return standards_text
    
    def _generate_review(self, pr_data: str, standards: str) -> str:
        """Генерирует финальное ревью кода"""
        
        prompt = f"""Ты опытный code reviewer. Проанализируй Pull Request.

# PR DATA
{pr_data}

# RELEVANT CODE STANDARDS
{standards}

Твоя задача:
1. Оценить улучшения в коде
2. Найти оставшиеся проблемы
3. Проверить соответствие стандартам
4. Дать конкретные рекомендации

Формат ответа:

## 📋 Executive Summary
[Краткая оценка PR в 2-3 предложениях]

## ✅ Improvements Made
[Что было улучшено в этом PR]

## ⚠️ Issues Found

### 🔴 Critical
[Критические проблемы]

### 🟡 Major
[Важные замечания]

### 🔵 Minor
[Мелкие улучшения]

## 💡 Specific Recommendations
[Конкретные советы с примерами кода]

## 📊 Code Quality Metrics
- **Security:** [1-10]
- **Code Style:** [1-10]
- **Maintainability:** [1-10]
- **Documentation:** [1-10]

## 🎯 Final Verdict
[✅ APPROVED / ⚠️ NEEDS CHANGES / ❌ REJECTED]
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        review_text = response.content[0].text
        
        print(f"  ✓ Review generated ({len(review_text)} chars)", file=sys.stderr)
        
        return review_text

def main():
    """Главная функция"""
    
    print("\n" + "="*70)
    print(" CODE REVIEW AGENT - LOCAL PR REVIEWER")
    print(" Using RAG for standards matching")
    print("="*70 + "\n")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ERROR: ANTHROPIC_API_KEY not set in .env file!")
        print("   Create .env file with: ANTHROPIC_API_KEY=your_key")
        return
    
    try:
        reviewer = PRReviewerAgent()
        review = reviewer.review_pr()
        
        print("\n" + "="*70)
        print("REVIEW RESULT")
        print("="*70 + "\n")
        print(review)
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()