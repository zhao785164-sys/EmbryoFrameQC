
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFile

POLICY = "strict_then_jpeg_truncated_fallback_v1"

class QCDecodeError(RuntimeError):
    def __init__(self, metadata):
        self.metadata = dict(metadata)
        super().__init__(
            metadata.get("fallback_error")
            or metadata.get("strict_error")
            or "decode failed"
        )

def _decode_once(data, allow_truncated, metadata):
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated
        with Image.open(BytesIO(data)) as source:
            metadata["image_format"] = source.format or ""
            metadata["source_mode"] = source.mode
            metadata["width"], metadata["height"] = source.size
            source.load()
            rgb = source.convert("RGB")
            rgb.load()
        return rgb
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

def decode_qc_rgb(data):
    metadata = {
        "decode_mode": "FAILED",
        "image_format": "",
        "source_mode": "",
        "width": "",
        "height": "",
        "strict_error": "",
        "fallback_error": "",
    }
    try:
        rgb = _decode_once(data, False, metadata)
    except Exception as error:
        metadata["strict_error"] = (
            type(error).__name__ + ": " + str(error)
        )
        eligible = (
            metadata["image_format"] == "JPEG"
            and isinstance(error, OSError)
            and "truncated" in str(error).lower()
        )
        if not eligible:
            raise QCDecodeError(metadata) from error
        try:
            rgb = _decode_once(data, True, metadata)
        except Exception as fallback_error:
            metadata["fallback_error"] = (
                type(fallback_error).__name__
                + ": " + str(fallback_error)
            )
            raise QCDecodeError(metadata) from fallback_error
        metadata["decode_mode"] = "TRUNCATED_ALLOWED"
    else:
        metadata["decode_mode"] = "STRICT_OK"
    return rgb, metadata

def load_qc_rgb(path):
    return decode_qc_rgb(Path(path).read_bytes())
