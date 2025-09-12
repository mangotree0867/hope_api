import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hope_api_db")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # File paths
        self.labels_csv_path = "/Users/mango/hope_api/labels.csv"
        self.word_list_csv_path = "/Users/mango/hope_api/SL_Partner_Word_List_01.csv"
        self.model_path = "/Users/mango/hope_api/model2/best_model_gemini.pth"
        
        # ML settings
        self.max_seq_length = 50
        self.num_features = 335
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

settings = Settings()