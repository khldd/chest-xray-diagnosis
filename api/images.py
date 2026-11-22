"""Upload validation: size limit and real image decoding (the file's content, not its header)."""
from __future__ import annotations

import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api import settings


def read_image(upload: UploadFile) -> Image.Image:
    """Decode an uploaded file into a fully loaded PIL image or raise a 4xx HTTPException."""
    data = upload.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large (limit {limit_mb} MB)")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # force decoding now so corrupt files fail here, not inside the model
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Not a valid image file") from exc
    return image
