# SIH 26034 Backend

FastAPI backend foundation for the AI-powered Legal Metrology compliance system.

## Prerequisites

- Python 3.11 or later

## Run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
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
