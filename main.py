from fastapi import FastAPI, File, UploadFile
import uvicorn
import io
from PIL import Image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "hello"}

@app.get("/ping")
def ping():
    return {"reply": "pong"}

@app.get("/greet")
def greet(name: str):
    return {"message": "hello " + name}

@app.post("/inspect")
def inspect(file: UploadFile = File(...)):
    data = file.file.read()
    image = Image.open(io.BytesIO(data))
    return {"width": image.width, "height": image.height, "mode": image.mode}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)