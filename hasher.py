import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Example usage
password = "1234"
hashed_password = hash_password(password)
print("Hashed Password for '1234':", hashed_password)
