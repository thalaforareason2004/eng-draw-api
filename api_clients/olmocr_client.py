import os
from typing import Any, Optional, Dict, List

from huggingface_hub import InferenceClient
from PIL import Image

MODEL_ID = "allenai/olmOCR-2-7B-1025"
HF_TOKEN_ENV = "HF_API_TOKEN"

_client: Optional[InferenceClient] = None


def get_client() -> InferenceClient:
    """Create and cache an InferenceClient for olmOCR."""
    global _client
    if _client is None:
        token = os.getenv(HF_TOKEN_ENV)
        if not token:
            raise RuntimeError(f"{HF_TOKEN_ENV} environment variable is not set.")
        _client = InferenceClient(model=MODEL_ID, token=token)
    return _client


def call_olmocr_raw(
    image: Image.Image,
    prompt: str = "Describe this engineering drawing page.",
    max_new_tokens: int = 256,
) -> Any:
    """
    Minimal call to olmOCR via InferenceClient.chat_completion.
    Returns the raw completion object so we can inspect it.
    """
    client = get_client()

    # Build a simple multi-modal chat message: one image + one text prompt.
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image": image},
                {"type": "input_text", "text": prompt},
            ],
        }
    ]

    completion = client.chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
    )

    return completion
