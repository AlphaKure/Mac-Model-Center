from pydantic import BaseModel

class BasicResponse(BaseModel):
    outline: str
    detail: str