import os
import hashlib
import secrets
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    login_id = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    is_valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserCreate(BaseModel):
    name: str
    login_id: str
    password: str
    email: EmailStr

class UserLogin(BaseModel):
    login_id: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    login_id: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class AuthService:
    @staticmethod
    def generate_salt() -> str:
        return secrets.token_hex(16)

    @staticmethod
    def hash_password(password: str, salt: str) -> str:
        password_salt = password + salt
        return hashlib.sha256(password_salt.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, salt: str, password_hash: str) -> bool:
        return AuthService.hash_password(password, salt) == password_hash

    @staticmethod
    def create_access_token(user_id: int, login_id: str) -> str:
        """간단한 랜덤 토큰 생성 (만료 없음)"""
        # 사용자 정보와 타임스탬프를 사용하여 보안 랜덤 토큰 생성
        token_data = f"{user_id}:{login_id}:{datetime.utcnow().isoformat()}:{secrets.token_urlsafe(32)}"
        return hashlib.sha256(token_data.encode()).hexdigest()

    @staticmethod
    def verify_token(token: str, db: Session) -> dict:
        """user_sessions 테이블에 토큰이 존재하는지 확인하여 토큰 검증"""
        # 이 토큰으로 세션 찾기
        session = db.query(UserSession).filter(
            UserSession.session_token == token,
            UserSession.is_valid == True
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        # 사용자 정보 가져오기
        user = db.query(User).filter(User.id == session.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return {
            "user_id": user.id,
            "login_id": user.login_id
        }

class TokenManager:
    """데이터베이스에서 세션 유효성을 확인하는 토큰 매니저"""

    def __init__(self):
        self.http_bearer = HTTPBearer()

    async def __call__(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
        db: Session = None
    ) -> dict:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code"
            )

        if not credentials.scheme == "Bearer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authentication scheme"
            )

        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session not provided"
            )

        # 데이터베이스를 사용하여 토큰 검증
        payload = AuthService.verify_token(credentials.credentials, db)
        return payload

token_manager = TokenManager()

def get_current_user_dependency(token_payload: dict = Depends(token_manager)) -> dict:
    return token_payload

class CurrentUserDependency:
    """
    현재 인증된 사용자를 가져오는 의존성 클래스.
    db를 매개변수로 받아 순환 임포트를 방지합니다.
    """
    def __init__(self):
        self.token_manager = token_manager

    def __call__(
        self,
        token_payload: dict = Depends(token_manager),
        db: Session = None
    ) -> User:
        from fastapi import HTTPException, status

        user_id = token_payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session not provided"
            )

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return user

# 임포트할 수 있는 인스턴스 생성
get_current_user = CurrentUserDependency()