from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from typing import Generator
import os
from app.config import settings

# Create the SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    echo=settings.debug
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine
)

Base = declarative_base()# Base class for our models

def get_db() -> Generator[Session, None,None]:
    db = SessionLocal()
    try:
        yield db
    finally: 
        db.close()

def init_db() -> None:
    """
    Initialize database tables.
    Creates all tables defined in models.
    """
    Base.metadata.create_all(bind=engine)
