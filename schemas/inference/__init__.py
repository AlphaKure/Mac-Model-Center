from typing import Literal

from pydantic import BaseModel

class Progress(BaseModel):
    result: str
    percentage: float
    status: Literal["success", "running", "failed"]

class ProgressResponse(Progress):

    request_id: str