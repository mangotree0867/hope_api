from pydantic import BaseModel
from typing import List, Dict, Union, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    image_sequence: List[str]  # List of Base64 encoded images
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class PredictionResponse(BaseModel):
    words: List[Dict[str, Union[str, float]]]
    sentence: str
    message: str

class VideoPredictionResponse(BaseModel):
    prediction: Dict[str, Union[str, float]]
    top_predictions: List[Dict[str, Union[str, float]]]
    summary: Dict[str, Union[int, float, str]]
    sentence: str

class ChatRecordResponse(BaseModel):
    id: int
    user_id: Optional[str]
    session_id: Optional[str]
    predicted_word: str
    confidence: float
    generated_sentence: Optional[str]
    input_type: str
    frame_count: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]

class VideoRecordResponse(BaseModel):
    id: int
    user_id: Optional[str]
    session_id: Optional[str]
    filename: str
    file_size: int
    file_extension: str
    duration: Optional[float]
    frame_count: Optional[int]
    is_processed: bool
    chat_record_id: Optional[int]
    created_at: datetime
    updated_at: Optional[datetime]