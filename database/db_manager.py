import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', 5432),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )

def create_session(session_id, user_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO user_sessions (session_id, user_id, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (session_id) DO NOTHING
        """,
        (session_id, user_id, json.dumps({}))
    )
    
    conn.commit()
    cursor.close()
    conn.close()

def save_message(session_id, role, content, metadata=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO chat_history (session_id, role, content, metadata)
        VALUES (%s, %s, %s, %s)
        """,
        (session_id, role, content, json.dumps(metadata or {}))
    )
    
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history(session_id, limit=10):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute(
        """
        SELECT role, content, timestamp, metadata
        FROM chat_history
        WHERE session_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (session_id, limit)
    )
    
    messages = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return list(reversed(messages))

def update_session_activity(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        UPDATE user_sessions
        SET last_active = CURRENT_TIMESTAMP
        WHERE session_id = %s
        """,
        (session_id,)
    )
    
    conn.commit()
    cursor.close()
    conn.close()
