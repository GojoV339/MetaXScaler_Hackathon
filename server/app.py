# server/app.py
import uvicorn
from app import app  # Re-export the main FastAPI app from root

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
