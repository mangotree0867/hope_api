from typing import List, Dict, Union, Optional
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_sequence: List[str]  # List of Base64 encoded images

class PredictionResponse(BaseModel):
    words: List[str]
    sentence: str
    session_id: int
    user_message_id: Optional[int] = None
    assistant_message_id: Optional[int] = None