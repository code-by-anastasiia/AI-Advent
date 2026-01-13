# Python Best Practices

## Error Handling

### Always Handle Specific Exceptions
```python
# Good
try:
    user_id = int(input_value)
except ValueError as e:
    logger.error(f"Invalid user ID format: {e}")
    return None

# Bad
try:
    user_id = int(input_value)
except:  # Too broad!
    pass
```

### Don't Silence Errors
```python
# Bad
try:
    risky_operation()
except:
    pass  # Error disappeared!

# Good
try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise  # Re-raise or handle properly
```

## Return Early Pattern

Avoid deep nesting by returning early.
```python
# Good - flat structure
def validate_user(user: dict) -> bool:
    if not user:
        return False
    
    if not user.get('email'):
        return False
    
    if not user.get('password'):
        return False
    
    return True

# Bad - nested structure
def validate_user(user: dict) -> bool:
    if user:
        if user.get('email'):
            if user.get('password'):
                return True
    return False
```

## Avoid Unnecessary else After return
```python
# Good
def check_age(age: int) -> bool:
    if age >= 18:
        return True
    return False

# Even better
def check_age(age: int) -> bool:
    return age >= 18

# Bad - unnecessary else
def check_age(age: int) -> bool:
    if age >= 18:
        return True
    else:
        return False
```

## Use List Comprehensions for Simple Transformations
```python
# Good
squares = [x**2 for x in range(10)]
filtered = [x for x in data if x > 0]

# Bad - verbose
squares = []
for x in range(10):
    squares.append(x**2)
```

## Mutable Default Arguments - DANGER!
```python
# WRONG - mutable default
def add_item(item, items=[]):
    items.append(item)
    return items

# Correct
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

## Testing Requirements

- Minimum 70% code coverage
- Write tests BEFORE fixing bugs (TDD)
- Use pytest fixtures for setup/teardown
- Test edge cases and error conditions
