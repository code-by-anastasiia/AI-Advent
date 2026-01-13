def register(email, password):
    user = {}
    user['email'] = email
    user['password'] = password
    save_to_db(user)
    return user

def save_to_db(user):
    # TODO: implement
    pass

def login(email, password):
    # TODO: implement
    pass

def validate_email(email):
    if '@' in email:
        return True
    return False