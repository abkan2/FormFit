# Entry point of service 

# backend/app/main.py
from fastapi import FastAPI
import uvicorn

from app.routes import router

app = FastAPI()


app.include_router(router, prefix="/api/v1")
@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI!"}



if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)