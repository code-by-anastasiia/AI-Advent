import bcrypt
import re

def register_user(email: str, password: str) -> dict:
    """
    Register new user with secure password hashing
    
    Args:
        email: User email address
        password: Plain text password (will be hashed)
    
    Returns:
        User dictionary with hashed password
    """
    if not validate_email(email):
        raise ValueError("Invalid email format")
    
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    
    hashed_password = bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt()
    )
    
    user = {
        'email': email,
        'password': hashed_password.decode('utf-8')
    }
    
    save_to_database(user)
    return user

def save_to_database(user: dict) -> None:
    """Save user to database"""
    # TODO: implement database logic
    pass

def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user credentials"""
    # TODO: implement authentication
    return False

def validate_email(email: str) -> bool:
    """Validate email format using regex"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))