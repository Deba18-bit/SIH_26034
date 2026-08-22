# SIH 26034 Backend

FastAPI backend foundation for the AI-powered Legal Metrology compliance system.

## Prerequisites

- Python 3.11–3.13 (PaddlePaddle does not currently provide a Python 3.14 wheel)

## Run locally

```sh
python3.13 -m venv .venv-ocr
source .venv-ocr/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Check its status with:

```sh
curl http://127.0.0.1:8000/api/health
```

## Upload a package image

The current ingestion endpoint accepts JPEG, PNG, and WebP images up to 10 MiB.
It validates that the file can be decoded and is at least 320×240 pixels; it does
not perform OCR or compliance checks. It also returns an engineering quality
assessment based on dimensions, sharpness, brightness, and OpenCV readability.
An image that needs recapture is marked `MANUAL_REVIEW_REQUIRED`; this is not a
legal determination. A normalized grayscale PNG derivative is retained for the
future OCR stage while the original upload remains unchanged.

```sh
curl -X POST http://127.0.0.1:8000/api/scans \
  -F "image=@/path/to/package-label.jpg"
```

Original uploads are stored locally in `data/scans/` during development. Configure
the location and validation thresholds with `SIH_STORAGE_DIR`,
`SIH_MAX_UPLOAD_SIZE_BYTES`, `SIH_MIN_IMAGE_WIDTH`, and `SIH_MIN_IMAGE_HEIGHT`.
Quality thresholds can be configured with `SIH_MIN_SHARPNESS_VARIANCE`,
`SIH_MIN_BRIGHTNESS`, `SIH_MAX_BRIGHTNESS`, and `SIH_PREPROCESSING_DENOISE`.

## OCR evidence

The backend uses CPU PaddleOCR on the processed grayscale derivative only. The
original upload remains unchanged. `POST /api/scans` includes `ocr.items`; each
item has text, confidence (0–1), `engine`, source metadata, and a bounding box
in `[left, top, right, bottom]` pixel coordinates relative to that exact
processed derivative. Low-confidence text is preserved. `ocr.status` is
`completed`, `no_text`, or `failed`; OCR evidence is not a compliance decision.

PaddleOCR models download when the first real OCR request is made. Unit tests use
a fake OCR engine and do not download models.

## Configuration

Settings are read from environment variables prefixed with `SIH_`. For example:

```sh
export SIH_ENVIRONMENT=production
export SIH_APP_NAME="Legal Metrology Compliance API"
```

## Tests

```sh
pytest
```
