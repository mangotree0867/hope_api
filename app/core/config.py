import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Hope API - Sign Language Recognition"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hope_api_db")

    # Google API
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # ML Model paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    LABELS_CSV_PATH: str = str(BASE_DIR / "labels.csv")
    WORD_LIST_CSV_PATH: str = str(BASE_DIR / "SL_Partner_Word_List_01.csv")
    MODEL_PATH: str = str(BASE_DIR / "models" / "best_model_gemini.pth")

    # ML Model settings
    MAX_SEQ_LENGTH: int = 50
    NUM_FEATURES: int = 335

    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5

    def __init__(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

settings = Settings()