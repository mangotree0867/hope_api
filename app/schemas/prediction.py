from typing import List, Dict, Union, Optional
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_sequence: List[str]  # List of Base64 encoded images

class PredictionResponse(BaseModel):
    words: List[Dict[str, Union[str, float]]]
    sentence: str
    message: str
    session_id: int
    user_message_id: Optional[int] = None
    assistant_message_id: Optional[int] = None

class VideoPredictionResponse(BaseModel):
    prediction: Dict[str, Union[str, float]]
    top_predictions: List[Dict[str, Union[str, float]]]
    summary: Dict[str, Union[int, float, str]]
    sentence: str