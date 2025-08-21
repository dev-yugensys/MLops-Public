import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.db import get_connection, init_db, log_request, get_request_stats

def test_db():
    print("Testing database operations...")
    
    # Initialize database
    init_db()
    print("Database initialized")
    
    # Test logging a request
    test_data = {"feature1": 1.0, "feature2": 2.0}
    request_id = log_request("test_user", "test_model", test_data)
    print(f"Logged request with ID: {request_id}")
    
    # Get stats
    stats = get_request_stats()
    print("\nCurrent stats:")
    print(f"Total requests: {stats['total']}")
    print(f"By status: {stats['by_status']}")
    print(f"By model: {stats['by_model']}")

if __name__ == "__main__":
    test_db()
