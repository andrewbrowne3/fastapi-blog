import psycopg2
import os
import sys
from pathlib import Path

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "blog_agent_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=os.getenv("DB_PORT", "5432")
    )

def run_all_migrations():
    """Run all migration files in order"""
    migrations_dir = Path(__file__).parent
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        print("No migration files found")
        return
    
    try:
        conn = get_db_connection()
        print(f"Connected to database: {os.getenv('DB_NAME', 'blog_agent_db')}")
        
        for migration_file in migration_files:
            run_migration(migration_file, conn)
        
        conn.close()
        print("\n🎉 All migrations completed successfully!")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 