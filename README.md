# OCR as an Agent

HTTP API service for optical character recognition (OCR) using Azure AI Vision, supporting both images and PDFs.

## Features

- **Image OCR**: Supports PNG, JPG, WebP, BMP, GIF, TIFF formats (auto-normalizes to PNG)
- **PDF OCR**: Renders each page and performs OCR on individual pages
- **Two Input Modes**:
  - File upload: `POST /ocr/file` (multipart form)
  - URL-based: `POST /ocr/url` (JSON with public URL)
- **Swagger UI**: Auto-generated docs at `/` (redirects to `/docs`)
- **Retry Logic**: Built-in exponential backoff for transient failures

## Requirements

- Python 3.13.12+
- Azure AI Vision endpoint and API key

## Local Development

### Setup without Docker

1. **Clone/prepare the repo:**
   ```bash
   cd ocr-as-a-agent
   ```

2. **Create virtual environment:**
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Azure credentials:**
   ```bash
   # Create .env file with:
   export VISION_ENDPOINT="https://your-resource.cognitiveservices.azure.com"
   export VISION_KEY="your-api-key"
   export VISION_LANGUAGE="zh-Hant"  # Optional, defaults to Traditional Chinese
   ```

5. **Run the server:**
   ```bash
   python main.py
   ```

   Server starts at `http://localhost:8000`

---

## Docker Usage

### Build and Run Locally

**Option 1: Using docker-compose (recommended)**

```bash
# Start the service
docker-compose up --build

# Service runs at http://localhost:8000
```

The `.env` file will be automatically loaded from your local directory.

**Option 2: Using docker directly**

```bash
# Build the image
docker build -t ocr-agent:latest .

# Run container
docker run -d \
  --name ocr-api \
  -p 8000:8000 \
  -e VISION_ENDPOINT="https://your-resource.cognitiveservices.azure.com" \
  -e VISION_KEY="your-api-key" \
  ocr-agent:latest
```

### Environment Variables

When running in Docker, pass these via `-e` flag or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_ENDPOINT` | Azure AI Vision endpoint URL | Required |
| `VISION_KEY` | Azure API key | Required |
| `VISION_API_VERSION` | API version | `2024-02-01` |
| `VISION_LANGUAGE` | OCR language hint | `zh-Hant` |
| `VISION_MODEL_VERSION` | Model version | `latest` |
| `PDF_RENDER_DPI` | PDF rendering quality | `200` |
| `PORT` | Server port | `8000` |

---

## API Examples

### File Upload OCR

```bash
curl -X POST "http://localhost:8000/ocr/file" \
  -F "file=@/path/to/image.png"
```

Response:
```json
{
  "filename": "image.png",
  "text": "Extracted text from image..."
}
```

### URL-based OCR

```bash
curl -X POST "http://localhost:8000/ocr/url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/document.pdf"}'
```

Response:
```json
{
  "source": "https://example.com/document.pdf",
  "text": "===== Page 1 / 3 =====\nExtracted text...\n===== Page 2 / 3 =====\n..."
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

---

## Development Notes

### Image Format Support

- **Input**: PNG, JPG, WebP, BMP, GIF, TIFF
- **PDF Support**: Multi-page rendering at configurable DPI
- **Normalization**: Non-PDF images are automatically converted to PNG before OCR for compatibility

### Error Handling

- 400: Invalid input, empty file, or URL download failure
- 500: OCR processing failure or missing credentials

### Docker Multi-stage Build

The Dockerfile uses a multi-stage build to minimize image size:
1. **Builder stage**: Installs all build dependencies and creates wheels
2. **Runtime stage**: Copies only compiled wheels and runtime dependencies (no build tools)

Final image size: ~500MB (includes Python 3.13.12-slim + dependencies)

---

## License

MIT
