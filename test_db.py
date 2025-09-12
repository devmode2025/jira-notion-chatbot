import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    connection = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    db_version = cursor.fetchone()
    print(f"✅ Connected to PostgreSQL: {db_version[0]}")
    
    # Test vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✅ PGVector extension enabled")
    
    cursor.close()
    connection.close()
    
except Exception as e:
    print(f"❌ Connection failed: {e}")