from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from yolo_model import run_yolo_on_page

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze_yolo")
async def analyze_yolo(file: UploadFile = File(...)):
    # Read uploaded file into a PIL image
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # Run YOLO
    result = run_yolo_on_page(image, conf_threshold=0.3)
    crops = result.get("crops", [])

    # Build JSON-friendly detections (no images)
    detections = []
    for c in crops:
        detections.append(
            {
                "cls_id": c["cls_id"],
                "cls_name": c["cls_name"],
                "conf": float(c["conf"]),
                "box": c["box"],  # [x1, y1, x2, y2]
            }
        )

    return JSONResponse({"num_detections": len(detections), "detections": detections})