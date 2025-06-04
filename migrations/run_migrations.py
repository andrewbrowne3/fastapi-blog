#!/usr/bin/env python3
"""
Database migration runner for the FastAPI blog application.
Run this script to set up the database schema.
"""

import os
import sys
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "blog_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password"),
        port=os.getenv("DB_PORT", "5432")
    )

def run_migration(migration_file: Path, conn):
    """Run a single migration file"""
    print(f"Running migration: {migration_file.name}")
    
    with open(migration_file, 'r') as f:
        sql_content = f.read()
    
    with conn.cursor() as cursor:
        cursor.execute(sql_content)
    
    conn.commit()
    print(f"✅ Migration {migration_file.name} completed successfully")

def run_all_migrations():
    """Run all migration files in order"""
    migrations_dir = Path(__file__).parent
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        print("No migration files found")
        return
    
    try:
        conn = get_db_connection()
        print(f"Connected to database: {os.getenv('DB_NAME', 'blog_db')}")
        
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

def create_admin_user():
    """Create a default admin user"""
    try:
        conn = get_db_connection()
        
        # Import User model to hash password
        sys.path.append(str(Path(__file__).parent.parent))
        from models.user import User
        
        admin_password_hash = User.hash_password("admin123")
        
        with conn.cursor() as cursor:
            # Check if admin user already exists
            cursor.execute("SELECT id FROM users WHERE username = %s", ("admin",))
            if cursor.fetchone():
                print("Admin user already exists")
                return
            
            # Create admin user
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, first_name, last_name, is_admin, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, ("admin", "admin@example.com", admin_password_hash, "Admin", "User", True, True))
        
        conn.commit()
        conn.close()
        
        print("✅ Admin user created successfully!")
        print("   Username: admin")
        print("   Password: admin123")
        print("   Email: admin@example.com")
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")

if __name__ == "__main__":
    print("🚀 Starting database migrations...")
    run_all_migrations()
    
    # Ask if user wants to create admin user
    create_admin = input("\nDo you want to create a default admin user? (y/n): ").lower().strip()
    if create_admin in ['y', 'yes']:
        create_admin_user() 