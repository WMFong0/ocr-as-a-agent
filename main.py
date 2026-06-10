#!/usr/bin/env python3
"""
HTTP OCR service for image/PDF input.

Input modes:
- POST /ocr/file: multipart upload
- POST /ocr/url: JSON payload with file URL

Azure AI Vision Image Analysis (Read OCR):
POST {endpoint}/computervision/imageanalysis:analyze?features=read&api-version=2024-02-01
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=int(os.getenv("WORKERS", "25")),
    )
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Final
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure AI Vision configuration.
VISION_ENDPOINT: Final[str] = os.getenv("VISION_ENDPOINT", "").rstrip("/")
VISION_KEY: Final[str] = os.getenv("VISION_KEY", "")
API_VERSION: Final[str] = os.getenv("VISION_API_VERSION", "2024-02-01")
LANGUAGE: Final[str] = os.getenv("VISION_LANGUAGE", "zh-Hant")
MODEL_VERSION: Final[str] = os.getenv("VISION_MODEL_VERSION", "latest")

# Proxy configuration for HTTP requests.
PROXY_URL: Final[str | None] = os.getenv("ASW_PROXY_URL")
if PROXY_URL:
    logger.info(f"Proxy configured: {PROXY_URL}")
else:
    logger.debug("No proxy URL configured (ASW_PROXY_URL not set)")

# File extension hints used for lightweight type detection fallback.
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
PDF_EXTS: set[str] = {".pdf"}

# FastAPI app entrypoint with Swagger UI at /docs.
app = FastAPI(title="OCR as an Agent", version="1.0.0", docs_url="/docs")

OcrLine = dict[str, Any]

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
    logger.debug(f"_is_pdf: Checking file detection - filename={filename}, content_type={content_type}, data_size={len(data)}")
    
    # Content type is the strongest signal when available.
    if content_type and "pdf" in content_type.lower():
        logger.info(f"_is_pdf: Detected PDF via content-type: {content_type}")
        return True

    # Filename extension is a secondary fallback signal.
    if filename and filename.lower().endswith(tuple(PDF_EXTS)):
        logger.info(f"_is_pdf: Detected PDF via filename extension: {filename}")
        return True

    # Binary signature check for PDF magic header.
    is_pdf_magic = data.startswith(b"%PDF-")
    if is_pdf_magic:
        logger.info("_is_pdf: Detected PDF via magic header")
    else:
        logger.debug("_is_pdf: Content detected as non-PDF")
    return is_pdf_magic


def _line_text(line: dict[str, Any]) -> str | None:
    """Return normalized OCR line text from known Azure fields."""
    text = line.get("text") or line.get("content")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _line_bbox(line: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return (left, top, right, bottom) from Azure OCR line geometry."""
    raw_bbox = line.get("boundingBox") or line.get("polygon")
    points: list[tuple[float, float]] = []

    if isinstance(raw_bbox, list):
        if raw_bbox and all(isinstance(v, (int, float)) for v in raw_bbox):
            coords = [float(v) for v in raw_bbox]
            points = list(zip(coords[0::2], coords[1::2]))
        else:
            for point in raw_bbox:
                if isinstance(point, dict):
                    x = point.get("x")
                    y = point.get("y")
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        points.append((float(x), float(y)))
                elif (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                    and isinstance(point[0], (int, float))
                    and isinstance(point[1], (int, float))
                ):
                    points.append((float(point[0]), float(point[1])))

    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _make_ocr_line(line: dict[str, Any]) -> OcrLine | None:
    """Create a sortable line record when text and geometry are available."""
    text = _line_text(line)
    bbox = _line_bbox(line)
    if not text or bbox is None:
        return None

    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return None

    return {
        "text": text,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
        "x_center": (left + right) / 2,
        "y_center": (top + bottom) / 2,
    }


def _median(values: list[float], default: float) -> float:
    """Return a median without adding a runtime dependency."""
    if not values:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _row_sort_key(line: OcrLine) -> tuple[float, float]:
    """Sort key for regular top-to-bottom, left-to-right line reading."""
    return line["top"], line["left"]


def _split_vertical_sections(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Split page lines into vertical sections before column detection.

    Full-width headings above a multi-column body should remain before the
    columns, so we first partition on large vertical whitespace.
    """
    if len(lines) <= 1:
        return [lines]

    ordered = sorted(lines, key=_row_sort_key)
    heights = [line["height"] for line in ordered]
    median_height = _median(heights, 12.0)
    gap_threshold = max(median_height * 2.4, 24.0)

    sections: list[list[OcrLine]] = []
    current: list[OcrLine] = [ordered[0]]
    current_bottom = ordered[0]["bottom"]

    for line in ordered[1:]:
        gap = line["top"] - current_bottom
        if gap > gap_threshold:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
        current_bottom = max(current_bottom, line["bottom"])

    sections.append(current)
    return sections


def _split_columns(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Split a vertical section into detected columns from left to right."""
    if len(lines) < 4:
        return [sorted(lines, key=_row_sort_key)]

    section_left = min(line["left"] for line in lines)
    section_right = max(line["right"] for line in lines)
    section_width = max(section_right - section_left, 1.0)
    median_width = _median([line["width"] for line in lines], section_width)

    # If several lines are nearly full width, this section is probably a
    # heading/list area rather than independent columns.
    wide_lines = [line for line in lines if line["width"] >= section_width * 0.72]
    if len(wide_lines) >= max(2, len(lines) // 3):
        return [sorted(lines, key=_row_sort_key)]

    ordered = sorted(lines, key=lambda line: (line["left"], line["top"]))
    left_threshold = max(median_width * 0.35, section_width * 0.10, 40.0)
    columns: list[list[OcrLine]] = [[ordered[0]]]
    column_lefts: list[float] = [ordered[0]["left"]]

    for line in ordered[1:]:
        if line["left"] - column_lefts[-1] > left_threshold:
            columns.append([line])
            column_lefts.append(line["left"])
        else:
            columns[-1].append(line)
            column_lefts[-1] = min(column_lefts[-1], line["left"])

    if len(columns) == 1:
        return [sorted(lines, key=_row_sort_key)]

    # Very small trailing clusters are often side notes/noise, not a real
    # document column. Keep normal row order if the split looks accidental.
    min_column_size = max(2, len(lines) // 12)
    if any(len(column) < min_column_size for column in columns):
        return [sorted(lines, key=_row_sort_key)]

    return [sorted(column, key=_row_sort_key) for column in columns]


def _sort_ocr_lines_by_layout(lines: list[OcrLine]) -> list[str]:
    """Return text in page layout order, with multi-column sections handled."""
    sorted_text: list[str] = []
    for section in _split_vertical_sections(lines):
        section_left = min(line["left"] for line in section)
        section_right = max(line["right"] for line in section)
        section_width = max(section_right - section_left, 1.0)
        buffered_lines: list[OcrLine] = []

        for line in sorted(section, key=_row_sort_key):
            is_full_width = len(section) >= 4 and line["width"] >= section_width * 0.72
            if is_full_width:
                for column in _split_columns(buffered_lines):
                    sorted_text.extend(buffered_line["text"] for buffered_line in column)
                buffered_lines = []
                sorted_text.append(line["text"])
            else:
                buffered_lines.append(line)

        for column in _split_columns(buffered_lines):
            sorted_text.extend(line["text"] for line in column)
    return sorted_text


def _collect_lines_from_result(result: dict[str, Any]) -> tuple[list[OcrLine], list[str]]:
    """Collect OCR lines, preserving a plain fallback order for no-geometry cases."""
    positioned: list[OcrLine] = []
    fallback: list[str] = []

    def add_line(line: Any) -> None:
        if not isinstance(line, dict):
            return
        text = _line_text(line)
        if not text:
            return
        fallback.append(text)
        positioned_line = _make_ocr_line(line)
        if positioned_line is not None:
            positioned.append(positioned_line)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "lines" and isinstance(value, list):
                    for line in value:
                        add_line(line)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return positioned, fallback


def _extract_text_from_result(result: dict[str, Any]) -> str:
    """Extract OCR text lines from Azure Image Analysis response payload.

    The API can return slightly different nested shapes depending on version
    and backend processing. This function attempts multiple known paths and
    then falls back to a recursive scan for `lines` collections.
    """
    logger.debug(f"_extract_text_from_result: Starting text extraction from result keys: {list(result.keys())}")
    positioned_lines, fallback_lines = _collect_lines_from_result(result)
    if positioned_lines and len(positioned_lines) == len(fallback_lines):
        lines_out = _sort_ocr_lines_by_layout(positioned_lines)
        final_text = "\n".join(lines_out).strip()
        logger.info(
            "_extract_text_from_result: Extracted %s positioned lines after layout sorting. Final text length: %s chars",
            len(lines_out),
            len(final_text),
        )
        return final_text

    # Collect final output lines in provider order if geometry is unavailable.
    lines_out: list[str] = []

    # Most responses place OCR data under `readResult`.
    read = result.get("readResult") or result.get("read") or {}
    blocks = read.get("blocks")

    # Primary schema path: readResult.blocks[].lines[].text
    if isinstance(blocks, list):
        logger.debug(f"_extract_text_from_result: Found {len(blocks)} blocks in readResult")
        for b in blocks:
            if not isinstance(b, dict):
                continue
            for ln in (b.get("lines") or []):
                if not isinstance(ln, dict):
                    continue
                t = ln.get("text")
                if isinstance(t, str) and t.strip():
                    lines_out.append(t.strip())
        logger.info(f"_extract_text_from_result: Extracted {len(lines_out)} lines from primary schema")

    # Alternate schema path: read.pages[].lines[].content|text
    if not lines_out:
        logger.debug("_extract_text_from_result: Primary schema found no lines, trying alternate schema")
        pages = read.get("pages")
        if isinstance(pages, list):
            logger.debug(f"_extract_text_from_result: Found {len(pages)} pages in alternate schema")
            for p in pages:
                if not isinstance(p, dict):
                    continue
                for ln in (p.get("lines") or []):
                    if not isinstance(ln, dict):
                        continue
                    t = ln.get("content") or ln.get("text")
                    if isinstance(t, str) and t.strip():
                        lines_out.append(t.strip())
            logger.info(f"_extract_text_from_result: Extracted {len(lines_out)} lines from alternate schema")

    # Last-resort fallback: recursively search nested payloads for line entries.
    if not lines_out:
        logger.debug("_extract_text_from_result: No lines found in schemas, using recursive fallback")

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
        logger.info(f"_extract_text_from_result: Recursive fallback extracted {len(lines_out)} lines")

    # Join extracted lines into final plain text block.
    final_text = "\n".join(lines_out).strip()
    logger.info(f"_extract_text_from_result: Final text length: {len(final_text)} chars, lines_count: {len(lines_out)}")
    return final_text


def _post_ocr_bytes(payload: bytes, session: Any = None, timeout_s: int = 180) -> dict[str, Any]:
    """Call Azure Vision OCR endpoint with retries and return JSON response.

    Retries are applied for throttling (429) and transient server-side failures
    (5xx) using capped exponential backoff. Retry behavior:
    - Attempt 1: immediate
    - Attempt 2: 2 seconds
    - Attempt 3: 4 seconds  
    - Attempt 4: 8 seconds
    - Attempt 5: 16 seconds
    """
    # Validate required credentials before issuing request.
    if not VISION_ENDPOINT or not VISION_KEY:
        logger.error("Missing VISION_ENDPOINT or VISION_KEY")
        raise RuntimeError("Missing VISION_ENDPOINT or VISION_KEY")

    # Build endpoint URL and query parameters for Read OCR feature.
    url = f"{VISION_ENDPOINT}/vision/v3.2/read/analyzeResults"
    # For POST, we use the analyze endpoint
    url = f"{VISION_ENDPOINT}/vision/v3.2/read/analyze"
    logger.info(f"_post_ocr_bytes: Posting OCR request to: {url} (payload size: {len(payload)} bytes)")
    
    params = {
        "version": "v3.2",
    }

    # Use key auth and send binary payload bytes.
    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream",
    }

    # Keep track of throttling attempts for diagnostics.
    retry_reasons: list[str] = []
    last_resp: requests.Response | None = None

    # Use requests session with proxy if configured
    req_session = requests.Session()
    if PROXY_URL:
        req_session.proxies = {
            "http": PROXY_URL,
            "https": PROXY_URL,
        }
        logger.debug(f"_post_ocr_bytes: Proxy configured for session: {PROXY_URL}")

    # Retry loop for transient HTTP responses.
    for attempt in range(1, 6):
        logger.debug(f"_post_ocr_bytes: OCR attempt {attempt}/5")
        try:
            resp = req_session.post(url, params=params, headers=headers, data=payload, timeout=timeout_s)
            last_resp = resp

            # Check for throttling or server errors that warrant retry.
            should_retry = False
            retry_reason = ""
            
            if resp.status_code == 429:
                should_retry = True
                retry_reason = "rate-limited (429)"
                retry_reasons.append(f"Attempt {attempt}: {retry_reason}")
                logger.warning(f"_post_ocr_bytes: Rate limited (429). Retrying after backoff...")
            elif 500 <= resp.status_code < 600:
                should_retry = True
                retry_reason = f"server error ({resp.status_code})"
                retry_reasons.append(f"Attempt {attempt}: {retry_reason}")
                logger.warning(f"_post_ocr_bytes: Server error {resp.status_code}. Retrying...")

            if should_retry:
                # Capped exponential backoff (max 20 seconds between attempts).
                backoff: int = min(2 ** attempt, 20)
                time.sleep(backoff)
                continue

            try:
                # Raise for non-success status codes.
                resp.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

            # Successful OCR response - for async API, get the operation URL
            if resp.status_code == 202:
                # Async operation started, get result from Operation-Location header
                operation_url = resp.headers.get("Operation-Location")
                if not operation_url:
                    raise RuntimeError("No Operation-Location header in response")
                logger.debug(f"_post_ocr_bytes: Async operation started, polling: {operation_url}")
                
                # Poll for result
                result = _poll_ocr_result(operation_url, VISION_KEY, PROXY_URL, timeout_s)
            else:
                # Synchronous result (shouldn't happen with current API, but handle it)
                result = resp.json()
            
            logger.info(f"_post_ocr_bytes: OCR succeeded on attempt {attempt}")
            
            # Include retry diagnostic info if retries occurred.
            if retry_reasons:
                result["_retry_history"] = retry_reasons
            
            return result
        
        except Exception as e:
            if attempt < 5:
                logger.debug(f"_post_ocr_bytes: Attempt {attempt} failed: {str(e)}")
                time.sleep(min(2 ** attempt, 20))
            else:
                raise

    # All retries exhausted - provide detailed error context.
    error_msg = f"Too many retries (5 attempts exhausted). Last HTTP {last_resp.status_code if last_resp else 'n/a'}"
    logger.error(error_msg)
    if retry_reasons:
        error_msg += f"\nRetry history: {'; '.join(retry_reasons)}"
    raise RuntimeError(error_msg)


def _poll_ocr_result(operation_url: str, api_key: str, proxy_url: str | None, timeout_s: int = 180) -> dict[str, Any]:
    """Poll the async OCR operation result until completion."""
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    poll_session = requests.Session()
    if proxy_url:
        poll_session.proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }
    
    max_polls = 60  # up to 60 polls = ~5 min if 5s intervals
    poll_count = 0
    
    while poll_count < max_polls:
        poll_count += 1
        resp = poll_session.get(operation_url, headers=headers, timeout=timeout_s)
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("status") == "succeeded":
                logger.info(f"_poll_ocr_result: Operation completed after {poll_count} polls")
                return result
            elif result.get("status") == "failed":
                raise RuntimeError(f"OCR operation failed: {result.get('analyzeResult', {})}")
        
        # Still processing, wait before next poll
        logger.debug(f"_poll_ocr_result: Poll {poll_count} - status: {result.get('status', 'unknown')}")
        time.sleep(1)
    
    raise RuntimeError(f"OCR operation timed out after {max_polls} polls")


def _ocr_payload(data: bytes, filename: str | None, content_type: str | None) -> str:
    """Dispatch payload to PDF or image OCR pipeline based on detected type."""
    # PDF render DPI is configurable via environment.
    dpi = int(os.getenv("PDF_RENDER_DPI", "200"))
    
    is_pdf = _is_pdf(filename, content_type, data)
    logger.info(f"Processing {'PDF' if is_pdf else 'image'} file: {filename or 'unknown'}")

    # Normalize non-PDF images (e.g. webp) into PNG bytes for broader OCR compatibility.
    if not is_pdf:
        data = _normalize_image_bytes(data)

    # Azure Vision client handles proxy through environment variables automatically.
    logger.debug("_ocr_payload: Using Azure Vision SDK (proxy configured via environment)")
    
    if is_pdf:
        return ocr_pdf_bytes(data, dpi=dpi)
    return ocr_image_bytes(data)


def _normalize_image_bytes(data: bytes) -> bytes:
    """Convert image bytes into PNG bytes to maximize OCR endpoint compatibility."""
    logger.debug(f"_normalize_image_bytes: Converting image (input size: {len(data)} bytes)")
    try:
        with Image.open(BytesIO(data)) as img:
            logger.debug(f"_normalize_image_bytes: Original image format={img.format}, mode={img.mode}, size={img.size}")
            rgb = img.convert("RGB")
            logger.debug(f"_normalize_image_bytes: Converted to RGB")
            buf = BytesIO()
            rgb.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            logger.info(f"_normalize_image_bytes: Conversion successful. PNG output size: {len(png_bytes)} bytes")
            return png_bytes
    except Exception as e:
        # If conversion fails, pass through original payload and let OCR endpoint decide.
        logger.warning(f"_normalize_image_bytes: Image conversion failed: {str(e)}. Passing through original data")
        return data
    
# ==================================
# Functions
# ==================================

def ocr_image_bytes(payload: bytes) -> str:
    """Run OCR for image bytes and return extracted text."""
    logger.info(f"ocr_image_bytes: Starting image OCR (payload size: {len(payload)} bytes)")
    result = _post_ocr_bytes(payload)
    logger.debug(f"ocr_image_bytes: Received OCR result, extracting text")
    text = _extract_text_from_result(result)
    logger.info(f"ocr_image_bytes: Image OCR complete. Extracted {len(text)} characters")
    return text


def pdf_bytes_to_page_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Render in-memory PDF bytes into per-page PNG bytes for OCR processing."""
    logger.info(f"pdf_bytes_to_page_images: Starting PDF rendering (input size: {len(pdf_bytes)} bytes, DPI: {dpi})")
    
    # PDF rendering path requires PyMuPDF.
    if fitz is None:
        logger.error("pdf_bytes_to_page_images: PyMuPDF not installed")
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

    # Open PDF directly from bytes and prepare output container.
    logger.debug("pdf_bytes_to_page_images: Opening PDF from bytes")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    logger.info(f"pdf_bytes_to_page_images: PDF opened successfully. Total pages: {page_count}")
    out: list[bytes] = []

    # PyMuPDF uses 72 DPI as the base coordinate system.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    logger.debug(f"pdf_bytes_to_page_images: Zoom factor: {zoom}")

    # Convert each page to PNG bytes so image OCR path can be reused.
    for i in range(page_count):
        logger.debug(f"pdf_bytes_to_page_images: Rendering page {i+1}/{page_count}")
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        out.append(png_bytes)
        logger.debug(f"pdf_bytes_to_page_images: Page {i+1} rendered to PNG ({len(png_bytes)} bytes)")

    # Explicit close to release document resources quickly.
    doc.close()
    logger.info(f"pdf_bytes_to_page_images: PDF rendering complete. Generated {len(out)} page images")
    return out


def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 200) -> str:
    """Run OCR for PDF bytes by rendering pages and concatenating page text."""
    logger.info(f"ocr_pdf_bytes: Starting PDF OCR (input size: {len(pdf_bytes)} bytes)")
    
    # Render each page to image bytes first.
    logger.debug("ocr_pdf_bytes: Rendering PDF pages to images")
    page_imgs = pdf_bytes_to_page_images(pdf_bytes, dpi=dpi)
    logger.info(f"ocr_pdf_bytes: PDF has {len(page_imgs)} pages to process")
    parts: list[str] = []

    # OCR page-by-page and prepend page headers for readability.
    for idx, img_bytes in enumerate(page_imgs, start=1):
        logger.info(f"ocr_pdf_bytes: Processing page {idx}/{len(page_imgs)} (image size: {len(img_bytes)} bytes)")
        result = _post_ocr_bytes(img_bytes)
        text = _extract_text_from_result(result).strip()
        logger.debug(f"ocr_pdf_bytes: Page {idx} OCR extracted {len(text)} characters")
        parts.append(f"===== Page {idx} / {len(page_imgs)} =====")
        parts.append(text)
        parts.append("")

    # Return merged output with trailing newline.
    final_text = "\n".join(parts).rstrip() + "\n"
    logger.info(f"ocr_pdf_bytes: PDF OCR complete. Total output: {len(final_text)} characters")
    return final_text

# ==================================
# API Routing
# ==================================

@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    """Redirect root path to Swagger docs."""
    logger.debug("GET /: Redirecting to /docs")
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    logger.debug("GET /health: Health check request")
    return {"status": "ok"}

@app.get("/test-connection")
def test_ms_connection() -> dict[str, Any]:
    """Test Connection with MS Azure Vision Service.
    
    Validates that required environment variables are set and performs
    a lightweight ping to the Azure Vision API endpoint.
    """
    logger.info("GET /test-connection: Starting Azure Vision connection test")
    result: dict[str, Any] = {"status": "ok", "details": {}}
    
    # Check required credentials.
    if not VISION_ENDPOINT:
        logger.error("test_ms_connection: VISION_ENDPOINT not configured")
        result["status"] = "error"
        result["error"] = "VISION_ENDPOINT not configured"
        return result
    
    if not VISION_KEY:
        logger.error("test_ms_connection: VISION_KEY not configured")
        result["status"] = "error"
        result["error"] = "VISION_KEY not configured"
        return result
    
    logger.debug("test_ms_connection: Credentials validated")
    
    # Record configuration (without exposing full key).
    result["details"]["endpoint"] = VISION_ENDPOINT
    result["details"]["api_version"] = API_VERSION
    result["details"]["language"] = LANGUAGE
    result["details"]["model_version"] = MODEL_VERSION
    logger.debug(f"test_ms_connection: Configuration - endpoint={VISION_ENDPOINT}, api_version={API_VERSION}")
    
    # Try a minimal request to validate endpoint connectivity.
    try:
        logger.debug(f"test_ms_connection: Testing connection to {VISION_ENDPOINT}")
        url = f"{VISION_ENDPOINT}/vision/v3.2/read/analyze"
        headers = {
            "Ocp-Apim-Subscription-Key": VISION_KEY,
            "Content-Type": "application/octet-stream",
        }
        
        # Create a 100x100 white test image with PIL (minimum size for Azure Vision API)
        from io import BytesIO
        test_image = Image.new("RGB", (100, 100), color="white")
        test_image_bytes = BytesIO()
        test_image.save(test_image_bytes, format="PNG")
        test_image_bytes.seek(0)
        test_png = test_image_bytes.getvalue()
        
        logger.debug(f"test_ms_connection: Sending test image ({len(test_png)} bytes) to {url}")
        test_session = requests.Session()
        if PROXY_URL:
            test_session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
        
        resp = test_session.post(url, headers=headers, data=test_png, timeout=30)
        logger.debug(f"test_ms_connection: Response status code: {resp.status_code}")
        
        if resp.status_code in (202, 200):
            logger.info("test_ms_connection: Successfully connected to Azure Vision API")
            result["status"] = "ok"
            result["message"] = "Successfully connected to Azure Vision API"
        elif resp.status_code in (401, 403):
            logger.error(f"test_ms_connection: Authentication failed ({resp.status_code})")
            result["status"] = "error"
            result["error"] = "Authentication failed: Invalid VISION_KEY or expired credentials"
        elif resp.status_code >= 500:
            logger.warning(f"test_ms_connection: Server error {resp.status_code}")
            result["status"] = "warning"
            result["warning"] = f"Server error {resp.status_code}: {resp.text[:100]}"
        else:
            logger.warning(f"test_msz_connection: Unexpected status {resp.status_code}")
            result["status"] = "warning"
            result["message"] = f"Unexpected response status {resp.status_code}: {resp.text[:100]}"
    
    except requests.exceptions.Timeout:
        logger.error("test_ms_connection: Connection timeout")
        result["status"] = "error"
        result["error"] = "Connection timeout to Azure Vision endpoint"
    except requests.exceptions.ConnectionError as e:
        logger.error(f"test_ms_connection: Connection error - {str(e)}")
        result["status"] = "error"
        result["error"] = "Failed to connect to Azure Vision endpoint"
    except Exception as e:
        logger.error(f"test_ms_connection: Error - {str(e)}")
        result["status"] = "error"
        result["error"] = f"Connection failed: {str(e)}"
    
    logger.info(f"test_ms_connection: Test complete. Status: {result.get('status')}")
    
    return result

@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...)) -> dict[str, str]:
    """Accept multipart file upload, run OCR, and return text output."""
    logger.info(f"POST /ocr/file: Received file upload - filename={file.filename}, content_type={file.content_type}")
    try:
        # Read full upload payload into memory.
        logger.debug("ocr_file: Reading uploaded file into memory")
        data = await file.read()
        logger.info(f"ocr_file: File read complete. Size: {len(data)} bytes")
        
        if not data:
            logger.error("ocr_file: Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Process uploaded bytes using shared OCR dispatcher.
        logger.debug("ocr_file: Dispatching to OCR payload processor")
        text = _ocr_payload(data, file.filename, file.content_type)
        logger.info(f"ocr_file: OCR processing complete. Output size: {len(text)} characters")
        return {"filename": file.filename or "unknown", "text": text}
    except HTTPException as e:
        # Preserve intentional HTTPException details/status.
        logger.warning(f"ocr_file: HTTP exception - {str(e)}")
        raise
    except Exception as e:
        # Normalize unexpected failures to 500 response.
        logger.error(f"ocr_file: Unexpected error - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ocr/url")
def ocr_url(req: UrlOCRRequest) -> dict[str, str]:
    """Download file from URL, run OCR, and return text output."""
    url_str = str(req.url)
    logger.info(f"POST /ocr/url: Received URL OCR request - url={url_str}")
    try:
        # Download target file bytes with timeout and browser-like defaults.
        logger.debug(f"ocr_url: Downloading file from URL (timeout: 60s)")
        request_headers = {
            # Browser Impersonation
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        download_session = requests.Session()
        if PROXY_URL:
            download_session.proxies = {
                "http": PROXY_URL,
                "https": PROXY_URL,
            }

        resp = download_session.get(
            url_str,
            timeout=60,
            headers=request_headers,
            allow_redirects=True,
        )
        logger.debug(f"ocr_url: Download response status: {resp.status_code}")

        if resp.status_code == 403:
            logger.warning("ocr_url: URL blocked by remote host (403), likely anti-bot/hotlink protection")
            raise HTTPException(
                status_code=400,
                detail=(
                    "Failed to download URL: remote host returned 403 (anti-bot/hotlink protection). "
                    "Try /ocr/file upload instead, or call /ocr/url with request headers/cookies "
                    "captured from a successful Postman request."
                ),
            )

        resp.raise_for_status()
        logger.info(f"ocr_url: Download successful. File size: {len(resp.content)} bytes")

        # Gather optional hints to improve type detection.
        content_type = resp.headers.get("Content-Type", "")
        filename = url_str.rsplit("/", 1)[-1] if "/" in url_str else None
        logger.debug(f"ocr_url: Extracted filename={filename}, content_type={content_type}")

        # Reuse common OCR dispatcher for URL payload.
        logger.debug(f"ocr_url: Dispatching to OCR payload processor")
        text = _ocr_payload(resp.content, filename, content_type)
        logger.info(f"ocr_url: OCR processing complete. Output size: {len(text)} characters")
        return {"source": url_str, "text": text}
    except requests.RequestException as e:
        logger.error(f"ocr_url: Failed to download URL - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}") from e
    except Exception as e:
        # Convert download/OCR failure into consistent API error response.
        logger.error(f"ocr_url: Unexpected error - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    """Run FastAPI app directly via Uvicorn in local development."""
    
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("uvicorn is required. Install with: pip install uvicorn") from e

    # Start the API server with configurable port.
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False, workers=25)
