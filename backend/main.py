# Entry point of service 

# backend/main.py
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI()


 # Allow frontend to connect (CORS for HTTP only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/api/v1")
@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI!"}



if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)