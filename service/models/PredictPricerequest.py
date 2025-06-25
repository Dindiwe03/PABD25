from pydantic import BaseModel

class PredictPriceRequest(BaseModel):
    area: float
    floor: int
    floors_count: int
    rooms_count: int