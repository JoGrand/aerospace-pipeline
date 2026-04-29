# API schemas
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    data: str