"""
Database initialization script.
Creates all tables defined in SQLAlchemy models.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.database import init_db, engine
from app.config import settings
from sqlalchemy import inspect, text


def check_database_connection():
    """Check if database connection is working."""
    print("🔌 Testing database connection...")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def list_existing_tables():
    """List all existing tables in the database."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if tables:
        print(f"\n📊 Existing tables ({len(tables)}):")
        for table in tables:
            print(f"  - {table}")
    else:
        print("\n📊 No existing tables found")
    
    return tables


def main():
    """Main initialization function."""
    print("=" * 60)
    print("🚀 Legal Document Analyzer - Database Initialization")
    print("=" * 60)
    print(f"\nEnvironment: {settings.environment}")
    print(f"Database: {settings.database_url.split('@')[1].split('/')[0]}\n")
    
    # Check database connection
    if not check_database_connection():
        print("\n⚠️  Please check your DATABASE_URL in .env file")
        sys.exit(1)
    
    # Import models (must import to register with Base)
    from app.auth.models import User
    from app.documents.models import Document
    
    # List existing tables
    existing_tables = list_existing_tables()
    
    # Initialize database (create tables)
    print("\n🔨 Creating database tables...")
    try:
        init_db()
        print("✅ Database tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        sys.exit(1)
    
    # List tables after initialization
    new_tables = list_existing_tables()
    
    # Show what was created
    created_tables = set(new_tables) - set(existing_tables)
    if created_tables:
        print(f"\n✨ New tables created:")
        for table in created_tables:
            print(f"  - {table}")
    else:
        print("\n✨ All tables already existed")
    
    print("\n" + "=" * 60)
    print("🎉 Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()