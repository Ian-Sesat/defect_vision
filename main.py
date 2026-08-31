import io
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from inspectors import ClipInspector

app = FastAPI()

recipe = yaml.safe_load(open("recipe.yaml"))
THRESHOLD = recipe["threshold"]
inspector = ClipInspector(recipe["ok"], recipe["defect"])


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/inspect")
def inspect(file: UploadFile = File(...)):
    data = file.file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file isn't a readable image.")

    score = inspector.defect_score(image)
    return {
        "verdict": "NOK" if score >= THRESHOLD else "OK",
        "defect_score": round(score, 4),
        "backend": inspector.name,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
