import pymysql

# -------------------------------
# MySQL Configuration
# -------------------------------
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "4566"  # your MySQL root password
DB_NAME = "auth_system"

# -------------------------------
# Connect to MySQL Server
# -------------------------------
def get_connection(create_db=True):
    """Return a connection to the MySQL database."""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
    except pymysql.err.OperationalError:
        if create_db:
            create_database()
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                cursorclass=pymysql.cursors.DictCursor
            )
        else:
            raise
    return conn

# -------------------------------
# Create Database
# -------------------------------
def create_database():
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.commit()
    conn.close()
    print(f"✅ Database '{DB_NAME}' is ready.")

# -------------------------------
# Create Tables
# -------------------------------
def create_users_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            user_type VARCHAR(50) NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Table 'users' is ready.")

# -------------------------------
# Initialize DB
# -------------------------------
def init_db():
    create_database()
    create_users_table()

if __name__ == "__main__":
    init_db()
