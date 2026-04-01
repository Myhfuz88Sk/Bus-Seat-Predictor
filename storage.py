import json
import os

FILE_NAME = "users.json"

def load_users():
    if not os.path.exists(FILE_NAME):
        return []
    with open(FILE_NAME, "r") as f:
        return json.load(f)

def save_users(users):
    with open(FILE_NAME, "w") as f:
        json.dump(users, f, indent=4)

def add_user(user):
    users = load_users()
    users.append(user)
    save_users(users)

def find_user_by_email(email):
    users = load_users()
    for user in users:
        if user["email"] == email:
            return user
    return None
