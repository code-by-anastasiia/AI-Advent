# Python Style Guide

## Type Hints - MANDATORY

All function signatures must include type hints.

### Good Examples
```python
def calculate_total(items: list[dict], tax_rate: float) -> float:
    pass

def get_user(user_id: int) -> dict | None:
    pass

def process_data(data: str) -> tuple[bool, str]:
    pass
```

### Bad Examples
```python
def calculate_total(items, tax_rate):  # Missing type hints
    pass
```

## Naming Conventions

### Functions
- Use `snake_case`
- Start with verb: `get_`, `create_`, `validate_`, `calculate_`
- Be descriptive, avoid abbreviations

Good: `register_user`, `validate_email`, `calculate_total_price`
Bad: `reg`, `val`, `calc`

### Classes
- Use `PascalCase`
- Noun or noun phrase

Good: `UserManager`, `DatabaseConnection`, `EmailValidator`
Bad: `user_manager`, `db_conn`, `validator`

### Constants
- Use `UPPER_SNAKE_CASE`

Good: `MAX_LOGIN_ATTEMPTS`, `DEFAULT_TIMEOUT`, `API_BASE_URL`

### Private Methods/Variables
- Prefix with single underscore `_`
```python
class User:
    def _hash_password(self, password: str) -> str:
        """Private method"""
        pass
```

## Docstrings - Required for Public Functions

Use Google or NumPy style docstrings.
```python
def register_user(email: str, password: str) -> dict:
    """
    Register a new user account.
    
    Args:
        email: User's email address
        password: Plain text password (will be hashed)
    
    Returns:
        Dictionary containing user data with hashed password
    
    Raises:
        ValueError: If email format is invalid or password too short
    """
    pass
```

## Code Organization

- Maximum function length: 50 lines
- Maximum line length: 88 characters (Black formatter default)
- One class per file (except small helper classes)