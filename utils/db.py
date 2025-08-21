import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import json

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "model_requests.db"
DB_PATH.parent.mkdir(exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT,
            input_data TEXT,
            output_data TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT
        )
        ''')
        conn.commit()

def log_request(user_id: str, model_name: str, input_data: Dict[str, Any]) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO model_requests (user_id, model_name, input_data) VALUES (?, ?, ?)',
            (user_id, model_name, json.dumps(input_data))
        )
        conn.commit()
        return cursor.lastrowid

def update_request(request_id: int, output_data: Dict[str, Any] = None, 
                 status: str = 'completed', error: str = None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE model_requests SET output_data=?, status=?, error_message=? WHERE id=?',
            (json.dumps(output_data) if output_data else None, status, error, request_id)
        )
        conn.commit()

def get_requests(user_id: str = None, limit: int = 100) -> List[Dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                'SELECT * FROM model_requests WHERE user_id=? ORDER BY timestamp DESC LIMIT ?',
                (user_id, limit)
            )
        else:
            cursor.execute('SELECT * FROM model_requests ORDER BY timestamp DESC LIMIT ?', (limit,))
        return [dict(row) for row in cursor.fetchall()]

def get_request_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM model_requests')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT status, COUNT(*) FROM model_requests GROUP BY status')
        status = dict(cursor.fetchall())
        cursor.execute('SELECT model_name, COUNT(*) FROM model_requests GROUP BY model_name')
        models = dict(cursor.fetchall())
        return {'total': total, 'by_status': status, 'by_model': models}

def delete_request(request_id: int) -> bool:
    """
    Delete a request from the database.
    
    Args:
        request_id: The ID of the request to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM model_requests WHERE id = ?', (request_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deleting request {request_id}: {str(e)}")
        return False

# Initialize database on import
init_db()
