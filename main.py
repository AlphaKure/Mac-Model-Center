import routers
from utils.config import read_config

from fastapi import FastAPI

SERVERHOST= read_config(section= "SERVERLET", key="host")
SERVERPORT = int(read_config(section= "SERVERLET", key="port"))

app = FastAPI(
    docs_url= "/api",
    version= "0.1.1"
)

app.include_router(routers.text2image_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host= SERVERHOST, port= SERVERPORT)