import routers

from fastapi import FastAPI

SERVERHOST= "0.0.0.0"
SERVERPORT = 8000

app = FastAPI(
    docs_url= "/api",
    version= "0.1.0"
)

app.include_router(routers.text2image_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host= SERVERHOST, port= SERVERPORT)