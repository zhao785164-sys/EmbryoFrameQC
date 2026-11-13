
import numpy as np
import torch
from PIL import Image, ImageOps

def prepare_qc(rgb):
    gray = rgb.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    gray.close()
    lo, hi = np.percentile(arr, [1, 99])
    flat = bool(hi <= lo)
    if flat:
        output = arr.astype(np.uint8)
    else:
        output = np.rint(
            np.clip((arr - lo) / (hi - lo), 0, 1) * 255
        ).astype(np.uint8)

    normalized = Image.fromarray(output)
    fitted = ImageOps.contain(
        normalized, (224, 224), method=Image.Resampling.BILINEAR
    )
    canvas = Image.new("L", (224, 224), 0)
    canvas.paste(
        fitted, ((224 - fitted.width) // 2, (224 - fitted.height) // 2)
    )
    display = canvas.convert("RGB")
    normalized.close()
    fitted.close()
    canvas.close()

    values = np.array(display, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, display, {
        "p01": float(lo), "p99": float(hi), "flat_percentiles": flat
    }
