from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

# weights/best.pt relative to this file
_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"

_yolo_model: Optional[YOLO] = None  # cached model


def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        if not _WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {_WEIGHTS_PATH}")
        _yolo_model = YOLO(str(_WEIGHTS_PATH))
    return _yolo_model


def run_yolo_on_page(image: Image.Image, conf_threshold: float = 0.3) -> Dict[str, Any]:
    """Run YOLO on full page and return annotated image + crops."""
    image = image.convert("RGB")
    img_rgb = np.array(image)

    model = get_yolo_model()
    results = model(img_rgb, conf=conf_threshold)[0]

    annotated = results.plot()  # RGB numpy array
    annotated_image = Image.fromarray(annotated)

    crops: List[Dict[str, Any]] = []
    names = results.names

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        else:
            cls_name = names[cls_id]

        if x2 <= x1 or y2 <= y1:
            continue

        crop_np = img_rgb[y1:y2, x1:x2]
        if crop_np.size == 0:
            continue

        crop_img = Image.fromarray(crop_np)

        crops.append(
            {
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": conf,
                "box": [x1, y1, x2, y2],
                "crop_image": crop_img,
            }
        )

    return {
        "annotated_image": annotated_image,
        "crops": crops,
    }
