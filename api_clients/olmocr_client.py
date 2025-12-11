import os
import io
import base64
from typing import Tuple, Any

import requests
from PIL import Image

MODEL_ID = "allenai/olmOCR-2-7B-1025"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HF_TOKEN_ENV = "HF_API_TOKEN"


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64," + b64


def call_olmocr_raw(
    image: Image.Image,
    prompt: str = "Describe this image.",
    max_new_tokens: int = 256,
) -> Tuple[int, Any]:
    """Call olmOCR via HF Inference API once; return (status_code, response_json_or_text)."""
    token = os.getenv(HF_TOKEN_ENV)
    if not token:
        raise RuntimeError(f"{HF_TOKEN_ENV} environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    img_url = _pil_to_data_url(image, fmt="PNG")

    payload = {
        "inputs": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "parameters": {"max_new_tokens": max_new_tokens},
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    return resp.status_code, data
