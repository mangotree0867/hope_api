from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.auth import (
    User, UserSession, UserCreate, UserLogin, TokenResponse, UserResponse,
    AuthService
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

def get_authenticated_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.
    Use this in any endpoint that requires authentication.
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )

    # Verify token and get user info
    token_payload = AuthService.verify_token(credentials.credentials, db)
    user_id = token_payload.get("user_id")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user

@router.post("/register", response_model=TokenResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user and return an access token
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.login_id == user_data.login_id).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this login_id already exists"
        )

    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    # Create new user
    salt = AuthService.generate_salt()
    password_hash = AuthService.hash_password(user_data.password, salt)

    new_user = User(
        name=user_data.name,
        login_id=user_data.login_id,
        email=user_data.email,
        password_hash=password_hash,
        salt=salt
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Generate access token
    access_token = AuthService.create_access_token(new_user.id, new_user.login_id)

    # Create session
    session = UserSession(
        user_id=new_user.id,
        session_token=access_token
    )
    db.add(session)
    db.commit()

    user_response = UserResponse(
        id=new_user.id,
        name=new_user.name,
        login_id=new_user.login_id,
        email=new_user.email,
        created_at=new_user.created_at
    )

    return TokenResponse(
        access_token=access_token,
        user=user_response
    )

@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login user and return an access token
    """
    # Find user by login_id
    user = db.query(User).filter(User.login_id == credentials.login_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials"
        )

    # Verify password
    if not AuthService.verify_password(credentials.password, user.salt, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials"
        )

    # Generate access token
    access_token = AuthService.create_access_token(user.id, user.login_id)

    # Create or update session
    existing_session = db.query(UserSession).filter(
        UserSession.user_id == user.id,
        UserSession.is_valid == True
    ).first()

    if existing_session:
        # Invalidate old session
        existing_session.is_valid = False
        db.commit()

    # Create new session
    new_session = UserSession(
        user_id=user.id,
        session_token=access_token
    )
    db.add(new_session)
    db.commit()

    user_response = UserResponse(
        id=user.id,
        name=user.name,
        login_id=user.login_id,
        email=user.email,
        created_at=user.created_at
    )

    return TokenResponse(
        access_token=access_token,
        user=user_response
    )

@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
):
    """
    Logout user by invalidating their session
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )

    # Find and invalidate the specific session token
    session = db.query(UserSession).filter(
        UserSession.session_token == credentials.credentials,
        UserSession.is_valid == True
    ).first()

    if session:
        session.is_valid = False
        db.commit()

    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_authenticated_user)
):
    """
    Get current authenticated user info
    """
    return UserResponse(
        id=current_user.id,
        name=current_user.name,
        login_id=current_user.login_id,
        email=current_user.email,
        created_at=current_user.created_at
    )