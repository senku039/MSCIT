# OCR Assets

This folder intentionally excludes large third-party installers and binaries.

## Policy
- Do **not** commit native OCR installers (for example `.exe` files) to Git.
- Store large binaries in release artifacts or external object storage.
- Keep only lightweight sample files needed for tests/demos.

## Tesseract installation
Install Tesseract from your OS package manager or official distribution and ensure
`tesseract` is available on PATH in runtime environments.
