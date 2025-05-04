import sqlite3
import os

def init_database():
    # Create database directory if it doesn't exist
    if not os.path.exists('database'):
        os.makedirs('database')

    # Connect to database
    conn = sqlite3.connect('database/users.db')
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin BOOLEAN DEFAULT 0
    )''')

    # Create access_logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER,
        access_type TEXT,
        status TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')

    # Create known_faces table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS known_faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        face_encoding BLOB,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')

    # Create default admin user if not exists
    cursor.execute('''
    INSERT OR IGNORE INTO users (username, password, is_admin)
    VALUES (?, ?, ?)
    ''', ('admin', 'admin123', True))

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_database()