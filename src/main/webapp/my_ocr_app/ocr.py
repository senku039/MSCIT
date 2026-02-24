"""Deprecated OCR app entrypoint kept for backward compatibility.

This module now reuses the primary Flask application so OCR behavior stays
consistent across deployment paths.
"""

from __future__ import annotations

from src.main.webapp.app import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
