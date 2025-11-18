"""
Authentication API routes.
Handles user registration, login, and user info endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import models, schemas, security

router = APIRouter()


@router.post("/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user.
    - **email**: Valid email address (must be unique)
    - **password**: Minimum 8 characters
    - **name**: Optional user name
    """
    # Check if user already exists
    existing_user = db.query(models.User).filter(
        models.User.email == user_data.email
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = security.hash_password(user_data.password)
    
    # Create new user
    new_user = models.User(
        email=user_data.email,
        password_hash=hashed_password,
        name=user_data.name
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=schemas.Token)
async def login(
    user_data: schemas.UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login and get access token.
    
    - **email**: User email
    - **password**: User password
    
    Returns JWT access token that expires in 24 hours.
    """
    # Find user by email
    user = db.query(models.User).filter(
        models.User.email == user_data.email
    ).first()
    
    # Verify user exists and password is correct
    if not user or not security.verify_password(user_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = security.create_access_token(
        data={"user_id": user.id, "email": user.email}
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/health")
async def auth_health_check():
    """Simple health check for auth service."""
    return {"status": "healthy", "service": "authentication"}