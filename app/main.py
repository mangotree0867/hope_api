import sys
import os
import warnings

# 모듈 임포트를 위해 현재 디렉토리를 파이썬 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# protobuf 의존 모듈을 임포트하기 전에 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

from fastapi import FastAPI
from app.core.config import settings
from app.core.database import engine
from app.api.routes import auth, predictions, chat

# 테이블 생성을 위한 모델 임포트
from app.models.auth import Base as AuthBase
from app.models.chat import Base as ChatBase

# 모든 테이블 생성
AuthBase.metadata.create_all(bind=engine)
ChatBase.metadata.create_all(bind=engine)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# 라우터 포함
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Sign Language Prediction API is running."}