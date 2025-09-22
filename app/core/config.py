import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Hope API - Sign Language Recognition"
    VERSION: str = "1.0.0"

    # 데이터베이스 설정
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hope_api_db")

    # Google API 설정
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # ML 모델 경로 설정
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SCALER_PATH: str = str(BASE_DIR / "scaler.pkl")
    LABEL_MAP_PATH: str = str(BASE_DIR / "label_map.json")
    WEIGHTS_PATH: str = str(BASE_DIR / "models" / "best_dynamic_features_model.pth")
    LABEL_WORD_LIST_PATH: str = str(BASE_DIR / "SL_Partner_Word_List_01.csv")

    def __init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

settings = Settings()