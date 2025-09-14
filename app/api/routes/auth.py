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
    현재 인증된 사용자를 가져오는 의존성.
    인증이 필요한 모든 엔드포인트에서 사용하세요.
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )

    # 토큰 검증 및 사용자 정보 가져오기
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
    새로운 사용자를 등록하고 액세스 토큰을 반환
    """
    # 사용자가 이미 존재하는지 확인
    existing_user = db.query(User).filter(User.login_id == user_data.login_id).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this login_id already exists"
        )

    # 이메일이 이미 존재하는지 확인
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )

    # 새로운 사용자 생성
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

    # 액세스 토큰 생성
    access_token = AuthService.create_access_token(new_user.id, new_user.login_id)

    # 세션 생성
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
    사용자 로그인 및 액세스 토큰 반환
    """
    # login_id로 사용자 찾기
    user = db.query(User).filter(User.login_id == credentials.login_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials"
        )

    # 비밀번호 검증
    if not AuthService.verify_password(credentials.password, user.salt, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials"
        )

    # 액세스 토큰 생성
    access_token = AuthService.create_access_token(user.id, user.login_id)

    # 세션 생성 또는 업데이트
    existing_session = db.query(UserSession).filter(
        UserSession.user_id == user.id,
        UserSession.is_valid == True
    ).first()

    if existing_session:
        # 이전 세션 무효화
        existing_session.is_valid = False
        db.commit()

    # 새로운 세션 생성
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
    세션을 무효화하여 사용자 로그아웃
    """
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme"
        )

    # 특정 세션 토큰 찾기 및 무효화
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
    현재 인증된 사용자 정보 가져오기
    """
    return UserResponse(
        id=current_user.id,
        name=current_user.name,
        login_id=current_user.login_id,
        email=current_user.email,
        created_at=current_user.created_at
    )