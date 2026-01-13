# Security Standards

## Password Handling

### Critical Rules
1. **Never store passwords in plain text**
   - Always use cryptographic hashing
   - Recommended: bcrypt, argon2, scrypt
   - Never use MD5 or SHA1 for passwords

2. **Password Requirements**
   - Minimum length: 8 characters
   - Recommend: mix of letters, numbers, symbols
   - Check against common password lists

3. **Salt Usage**
   - Each password must have unique salt
   - bcrypt handles this automatically
   - Never reuse salts

### Example - Good Password Hashing
```python
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')
```

### Example - Bad (Never do this!)
```python
# WRONG - plain text storage
user['password'] = password

# WRONG - weak hashing
user['password'] = hashlib.md5(password.encode()).hexdigest()
```

## Input Validation

### Email Validation
- Use regex patterns or validation libraries
- Check format before processing
- Sanitize input to prevent injection

### SQL Injection Prevention
- Use parameterized queries only
- Never concatenate user input into SQL
- Use ORM (SQLAlchemy) when possible

## Error Handling
- Don't reveal sensitive info in error messages
- Log security events (failed login attempts)
- Use generic messages for authentication errors
