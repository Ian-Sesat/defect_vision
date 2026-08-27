from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from fastapi.responses import FileResponse

import uvicorn
import io
import open_clip
import torch

app = FastAPI()

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

texts = tokenizer([
    "a photo of a clean metal surface",
    "a photo of a scratched metal surface",
])

with torch.no_grad():
    text_features = model.encode_text(texts)
text_features /= text_features.norm(dim=-1, keepdim=True)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/ping")
def ping():
    return {"reply": "pong"}

@app.get("/greet")
def greet(name: str):
    return {"message": "hello " + name}

@app.post("/inspect")
def inspect(file: UploadFile = File(...)):
    data = file.file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file isn't a readable image.")
    MIN_SIDE = 224
    if min(image.width, image.height) < MIN_SIDE:
        raise HTTPException(422, f"Image is too small to inspect. Minimum {MIN_SIDE} px on the short side.")
    MAX_MB = 50
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_MB} MB.")

    with torch.no_grad():
        image_features = model.encode_image(preprocess(image).unsqueeze(0))
    image_features /= image_features.norm(dim=-1, keepdim=True)

    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    defect_score = probs[0][1].item()

    return {
        "verdict": "NOK" if defect_score >= 0.5 else "OK",
        "defect_score": round(defect_score, 4),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
