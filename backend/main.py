# Entry point of service 

# backend/main.py
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from app.routes import router



app.include_router(router, prefix="/api/v1")
@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI!"}



if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)