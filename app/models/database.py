from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.sql import func
from ..database import Base

class ChatRecord(Base):
    __tablename__ = "chat_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    predicted_word = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    generated_sentence = Column(Text, nullable=True)
    input_type = Column(String, nullable=False)  # "video" or "image_sequence"
    frame_count = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class VideoRecord(Base):
    __tablename__ = "video_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)  # Path to stored file
    file_extension = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    is_processed = Column(Boolean, default=False)
    chat_record_id = Column(Integer, nullable=True)  # Link to prediction result
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())