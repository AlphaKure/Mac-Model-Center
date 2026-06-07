from typing import Literal
from pydantic import BaseModel

class RequestArgs(BaseModel, extra= "allow"):
    prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    output_path: str

class LoadModelArgs(BaseModel):
    model_path: str
    dtype: Literal["float16", "bfloat16"]