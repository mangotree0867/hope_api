import sys
import os
import warnings

# Add current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings early before importing protobuf-dependent modules
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

from fastapi import FastAPI
from app.core.config import settings
from app.core.database import engine
from app.api.routes import auth, predictions, chat

# Import models to create tables
from app.models.auth import Base as AuthBase
from app.models.chat import Base as ChatBase

# Create all tables
AuthBase.metadata.create_all(bind=engine)
ChatBase.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Sign Language Prediction API is running."}