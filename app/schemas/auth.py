from pydantic import BaseModel

class LogoutResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    detail: str