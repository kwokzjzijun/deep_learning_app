from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_to_tensor_file(image_path: str, out_path: str, image_size: int = 224) -> str:
    image = Image.open(image_path).convert("RGB")
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    image = image.resize((int(image_size), int(image_size)), resample)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    chw = np.transpose(arr, (2, 0, 1)).astype("<f4", copy=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(chw.tobytes(order="C"))
    return str(out)
