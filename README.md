# Dermalens — Skin-Lesion ML Research Sandbox

A privacy-first Flask application that validates skin-lesion images for a
computer-vision research pipeline and documents the model's limitations.

> **Not a medical device.** This application does not classify lesions, provide
> a diagnosis, or replace professional care. Do not use it to make health decisions.

## Live application

The Render service is prepared in [`render.yaml`](render.yaml). The verified
production URL will be added only after the repository is connected to Render.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FParthu-M%2FSkin_Cancer_Prediction)

## Screenshots

The interface includes a client-generated synthetic quality sample so the
workflow can be demonstrated without distributing or hotlinking patient imagery.

![Dermalens desktop research workspace](assets/dermalens-desktop.png)

### Mobile layout

![Dermalens responsive mobile layout](assets/dermalens-mobile.png)

## Why inference is intentionally disabled

The original repository referenced a local `swin_transformer_skin_cancer.pth`
file that was never committed and claimed 97% accuracy without a reproducible
evaluation. Shipping random or replacement weights would create a fake and
potentially dangerous medical result.

This version keeps the documented Swin Transformer research context while
deploying only functionality that can be verified:

- Real in-memory image decoding and content validation
- Resolution, exposure, contrast, and edge-detail checks
- Model-readiness report with measurable characteristics
- Explicit model card, seven-label research taxonomy, and known limitations
- No medical classification and no unsupported performance claim

## Product highlights

- Responsive drag-and-drop research interface
- JPEG, PNG, and WebP validation by declared and decoded formats
- 8 MB request limit, 20 megapixel limit, and decompression-bomb protection
- Image processing entirely in memory—uploads are never saved
- Strict Content Security Policy and privacy-focused response headers
- Health check and JSON model-card endpoints
- Gunicorn, Docker, and Render Blueprint production configuration
- Automated tests, coverage threshold, Ruff linting, and GitHub Actions CI

## Architecture

```text
.
├── app.py                 # Flask factory, API, model card, security headers
├── image_analysis.py      # safe image decoding and quality measurements
├── static/                # responsive interface and upload interactions
├── templates/             # semantic server-rendered pages
├── tests/                 # upload, validation, safety, and route tests
├── Dockerfile
├── gunicorn.conf.py
└── render.yaml
```

## API

### `GET /health`

Returns service health and the explicit `inference_enabled: false` state.

### `GET /api/model-card`

Returns the intended architecture, documented target labels, and limitations.

### `POST /api/analyze`

Accepts multipart form data with an `image` field. Successful responses contain:

- Decoded format and dimensions
- Resolution, exposure, contrast, and edge-detail checks
- Normalized quality metrics
- Readiness state
- An explicit statement that no inference was performed

## Local development

Requires Python 3.12.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt
python app.py
```

Open `http://127.0.0.1:5000`. Environment variables are documented in
[`.env.example`](.env.example); no secrets are required.

## Validation

```bash
ruff check .
pytest --cov --cov-report=term-missing
python -m compileall -q app.py image_analysis.py
docker build -t dermalens .
```

## Render deployment

1. Create a Render Blueprint from this repository.
2. Review the service described by [`render.yaml`](render.yaml).
3. Deploy and verify `/health`.

## Security and privacy decisions

- Uploaded bytes are never written to disk.
- Filename paths, incorrect MIME types, unreadable formats, small images, and
  excessive dimensions are rejected.
- API responses use `Cache-Control: no-store`.
- The interface instructs users to avoid identifying patient information.
- Production errors are generic and details remain in server logs.

## Future work

- Publish a versioned, checksummed model artifact only after reproducible training
- Document dataset license, split strategy, class imbalance, and external validation
- Add calibration and out-of-distribution evaluation
- Complete a formal model-risk review before considering any inference endpoint

## Author

**Majjiga Parthu** — M.Tech Computer Science and Engineering

[GitHub](https://github.com/Parthu-M) ·
[Portfolio](https://parthu-m.github.io/Portfolio/) ·
[LinkedIn](https://www.linkedin.com/in/majjigaparthu/)

## License

[MIT](LICENSE)
