#!/usr/bin/env python3
"""
HTTP OCR service for image/PDF input.

Input modes:
- POST /ocr/file: multipart upload
- POST /ocr/url: JSON payload with file URL

Azure AI Vision Image Analysis (Read OCR):
POST {endpoint}/computervision/imageanalysis:analyze?features=read&api-version=2024-02-01
"""

from __future__ import annotations

import os
import time
from typing import Any, Final

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
import requests

# PyMuPDF is optional at import time so non-PDF OCR still works when missing.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


# ==================================
# Config
# ==================================

# Load environment values from .env for local development.
load_dotenv(dotenv_path=".env")

# Azure AI Vision configuration.
VISION_ENDPOINT: Final[str] = os.getenv("VISION_ENDPOINT", "").rstrip("/")
VISION_KEY: Final[str] = os.getenv("VISION_KEY", "")
API_VERSION: Final[str] = os.getenv("VISION_API_VERSION", "2024-02-01")
LANGUAGE: Final[str] = os.getenv("VISION_LANGUAGE", "zh-Hant")
MODEL_VERSION: Final[str] = os.getenv("VISION_MODEL_VERSION", "latest")

# File extension hints used for lightweight type detection fallback.
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
PDF_EXTS: set[str] = {".pdf"}

# FastAPI app entrypoint with Swagger UI at /docs.
app = FastAPI(title="OCR as an Agent", version="1.0.0", docs_url="/docs")

# ==================================
# model
# ==================================

class UrlOCRRequest(BaseModel):
    """Request body model for URL-based OCR."""

    # Publicly accessible file URL to download and OCR.
    url: HttpUrl

# ==================================
# Helper function
# ==================================

def _is_pdf(filename: str | None, content_type: str | None, data: bytes) -> bool:
    """Detect whether provided payload should be treated as PDF."""
    # Content type is the strongest signal when available.
    if content_type and "pdf" in content_type.lower():
        return True

    # Filename extension is a secondary fallback signal.
    if filename and filename.lower().endswith(tuple(PDF_EXTS)):
        return True

    # Binary signature check for PDF magic header.
    return data.startswith(b"%PDF-")


def _extract_text_from_result(result: dict[str, Any]) -> str:
    """Extract OCR text lines from Azure Image Analysis response payload.

    The API can return slightly different nested shapes depending on version
    and backend processing. This function attempts multiple known paths and
    then falls back to a recursive scan for `lines` collections.
    """
    # Collect final output lines in reading order as best as possible.
    lines_out: list[str] = []

    # Most responses place OCR data under `readResult`.
    read = result.get("readResult") or result.get("read") or {}
    blocks = read.get("blocks")

    # Primary schema path: readResult.blocks[].lines[].text
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            for ln in (b.get("lines") or []):
                if not isinstance(ln, dict):
                    continue
                t = ln.get("text")
                if isinstance(t, str) and t.strip():
                    lines_out.append(t.strip())

    # Alternate schema path: read.pages[].lines[].content|text
    if not lines_out:
        pages = read.get("pages")
        if isinstance(pages, list):
            for p in pages:
                if not isinstance(p, dict):
                    continue
                for ln in (p.get("lines") or []):
                    if not isinstance(ln, dict):
                        continue
                    t = ln.get("content") or ln.get("text")
                    if isinstance(t, str) and t.strip():
                        lines_out.append(t.strip())

    # Last-resort fallback: recursively search nested payloads for line entries.
    if not lines_out:

        def walk(node: Any) -> None:
            """Recursively visit nested dict/list structures to find text lines."""
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "lines" and isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                t = item.get("text") or item.get("content")
                                if isinstance(t, str) and t.strip():
                                    lines_out.append(t.strip())
                    else:
                        walk(v)
            elif isinstance(node, list):
                for it in node:
                    walk(it)

        walk(result)

    # Join extracted lines into final plain text block.
    return "\n".join(lines_out).strip()


def _post_ocr_bytes(payload: bytes, session: requests.Session, timeout_s: int = 180) -> dict[str, Any]:
    """Call Azure OCR endpoint with retries and return JSON response.

    Retries are applied for throttling (429) and transient server-side failures
    (5xx) using capped exponential backoff.
    """
    # Validate required credentials before issuing request.
    if not VISION_ENDPOINT or not VISION_KEY:
        raise RuntimeError(
            "Missing VISION_ENDPOINT or VISION_KEY env var.\n"
            "Example:\n"
            "  export VISION_ENDPOINT='https://sa-st-mk4uzatu-eastus2.cognitiveservices.azure.com'\n"
            "  export VISION_KEY='...'\n"
        )

    # Build endpoint URL and query parameters for Read OCR feature.
    url = f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
    params = {
        "features": "read",
        "api-version": API_VERSION,
        "language": LANGUAGE,
        "model-version": MODEL_VERSION,
    }

    # Use key auth and send binary payload bytes.
    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream",
    }

    # Keep last response to improve final error detail when retries are exhausted.
    last_resp: requests.Response | None = None

    # Retry loop for transient HTTP responses.
    for attempt in range(1, 6):
        resp = session.post(url, params=params, headers=headers, data=payload, timeout=timeout_s)
        last_resp = resp

        # Retry when throttled or when server fails.
        if resp.status_code in (429,) or (500 <= resp.status_code < 600):
            # Capped exponential backoff.
            time.sleep(min(2 ** attempt, 20))
            continue

        try:
            # Raise for non-success status codes.
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

        # Successful OCR response.
        return resp.json()

    raise RuntimeError(f"Too many retries. Last HTTP {last_resp.status_code if last_resp else 'n/a'}")


def _ocr_payload(data: bytes, filename: str | None, content_type: str | None) -> str:
    """Dispatch payload to PDF or image OCR pipeline based on detected type."""
    # PDF render DPI is configurable via environment.
    dpi = int(os.getenv("PDF_RENDER_DPI", "200"))

    # Use a request-scoped session for API calls.
    with requests.Session() as session:
        if _is_pdf(filename, content_type, data):
            return ocr_pdf_bytes(data, session=session, dpi=dpi)
        return ocr_image_bytes(data, session=session)
    
# ==================================
# Functions
# ==================================

def ocr_image_bytes(payload: bytes, session: requests.Session) -> str:
    """Run OCR for image bytes and return extracted text."""
    result = _post_ocr_bytes(payload, session=session)
    return _extract_text_from_result(result)


def pdf_bytes_to_page_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Render in-memory PDF bytes into per-page PNG bytes for OCR processing."""
    # PDF rendering path requires PyMuPDF.
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    # Open PDF directly from bytes and prepare output container.
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[bytes] = []

    # PyMuPDF uses 72 DPI as the base coordinate system.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Convert each page to PNG bytes so image OCR path can be reused.
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out.append(pix.tobytes("png"))

    # Explicit close to release document resources quickly.
    doc.close()
    return out


def ocr_pdf_bytes(pdf_bytes: bytes, session: requests.Session, dpi: int = 200) -> str:
    """Run OCR for PDF bytes by rendering pages and concatenating page text."""
    # Render each page to image bytes first.
    page_imgs = pdf_bytes_to_page_images(pdf_bytes, dpi=dpi)
    parts: list[str] = []

    # OCR page-by-page and prepend page headers for readability.
    for idx, img_bytes in enumerate(page_imgs, start=1):
        result = _post_ocr_bytes(img_bytes, session=session)
        text = _extract_text_from_result(result).strip()
        parts.append(f"===== Page {idx} / {len(page_imgs)} =====")
        parts.append(text)
        parts.append("")

    # Return merged output with trailing newline.
    return "\n".join(parts).rstrip() + "\n"

# ==================================
# API Routing
# ==================================

@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect root path to Swagger docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...)) -> dict[str, str]:
    """Accept multipart file upload, run OCR, and return text output."""
    try:
        # Read full upload payload into memory.
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Process uploaded bytes using shared OCR dispatcher.
        text = _ocr_payload(data, file.filename, file.content_type)
        return {"filename": file.filename or "unknown", "text": text}
    except HTTPException:
        # Preserve intentional HTTPException details/status.
        raise
    except Exception as e:
        # Normalize unexpected failures to 500 response.
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ocr/url")
def ocr_url(req: UrlOCRRequest) -> dict[str, str]:
    """Download file from URL, run OCR, and return text output."""
    try:
        # Download target file bytes with timeout.
        resp = requests.get(str(req.url), timeout=60)
        resp.raise_for_status()

        # Gather optional hints to improve type detection.
        content_type = resp.headers.get("Content-Type", "")
        url_str = str(req.url)
        filename = url_str.rsplit("/", 1)[-1] if "/" in url_str else None

        # Reuse common OCR dispatcher for URL payload.
        text = _ocr_payload(resp.content, filename, content_type)
        return {"source": url_str, "text": text}
    except Exception as e:
        # Convert download/OCR failure into consistent API error response.
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    """Run FastAPI app directly via Uvicorn in local development."""
    
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("uvicorn is required. Install with: pip install uvicorn") from e

    # Start the API server with configurable port.
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
