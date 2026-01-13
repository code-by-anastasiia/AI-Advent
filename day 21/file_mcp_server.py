import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Все логи в stderr!
mcp = FastMCP("Local File PR Server")

@mcp.tool()
def get_pr_description() -> str:
    """
    Получает описание Pull Request из файла
    
    Returns:
        Текст описания PR
    """
    try:
        print("Reading PR description...", file=sys.stderr)
        
        desc_path = Path("mock_pr/pr_description.txt")
        
        if not desc_path.exists():
            return "ERROR: PR description file not found"
        
        description = desc_path.read_text(encoding='utf-8')
        
        print("PR description loaded successfully", file=sys.stderr)
        return description
        
    except Exception as e:
        error_msg = f"Error reading PR description: {str(e)}"
        print(error_msg, file=sys.stderr)
        return f"ERROR: {error_msg}"

@mcp.tool()
def get_code_diff() -> str:
    """
    Создаёт diff между старой и новой версией кода
    
    Returns:
        Diff в формате unified diff
    """
    try:
        print("Generating code diff...", file=sys.stderr)
        
        old_path = Path("mock_pr/old_version.py")
        new_path = Path("mock_pr/new_version.py")
        
        if not old_path.exists() or not new_path.exists():
            return "ERROR: Code files not found"
        
        old_code = old_path.read_text(encoding='utf-8')
        new_code = new_path.read_text(encoding='utf-8')
        
        # Простой diff
        result = "# Code Changes\n\n"
        result += "## Old Version (Before PR)\n"
        result += f"```python\n{old_code}\n```\n\n"
        result += "## New Version (After PR)\n"
        result += f"```python\n{new_code}\n```\n"
        
        print("Diff generated successfully", file=sys.stderr)
        return result
        
    except Exception as e:
        error_msg = f"Error generating diff: {str(e)}"
        print(error_msg, file=sys.stderr)
        return f"ERROR: {error_msg}"

@mcp.tool()
def get_old_code() -> str:
    """
    Получает старую версию кода
    
    Returns:
        Содержимое old_version.py
    """
    try:
        print("Reading old code version...", file=sys.stderr)
        
        path = Path("mock_pr/old_version.py")
        
        if not path.exists():
            return "ERROR: Old code file not found"
        
        code = path.read_text(encoding='utf-8')
        
        return f"# Old Version\n\n```python\n{code}\n```"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

@mcp.tool()
def get_new_code() -> str:
    """
    Получает новую версию кода (из PR)
    
    Returns:
        Содержимое new_version.py
    """
    try:
        print("Reading new code version...", file=sys.stderr)
        
        path = Path("mock_pr/new_version.py")
        
        if not path.exists():
            return "ERROR: New code file not found"
        
        code = path.read_text(encoding='utf-8')
        
        return f"# New Version (PR)\n\n```python\n{code}\n```"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    mcp.run()