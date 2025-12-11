from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path
import base64

# Ensure this directory is on sys.path so we can import yolo_model
sys.path.append(str(Path(__file__).resolve().parent))

from yolo_model import run_yolo_on_page

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/analyze_yolo")
async def analyze_yolo(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    result = run_yolo_on_page(image, conf_threshold=0.3)
    annotated = result.get("annotated_image")
    crops = result.get("crops", [])

    # Convert annotated PIL image to base64 PNG string
    annotated_b64 = None
    if annotated is not None:
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        buf.seek(0)
        annotated_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

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

    return JSONResponse(
        {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_b64,
        }
    )
