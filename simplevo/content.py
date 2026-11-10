"""Content items exchanged with the model: plain dicts, independent of any model provider.

``{"type": "text", "text": str}`` or ``{"type": "image", "png": bytes}``. A backend converts them to its own wire
format (see ``simplevo.backends``).
"""
from __future__ import annotations

import base64
import io
import json

from PIL import Image


def text(value) -> dict:
    return {"type": "text", "text": str(value)}


def as_json(value) -> dict:
    return text(json.dumps(value, ensure_ascii=False))


def image(png: bytes) -> dict:
    return {"type": "image", "png": bytes(png)}


def encode_png(picture: Image.Image) -> bytes:
    buffer = io.BytesIO()
    picture.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()
