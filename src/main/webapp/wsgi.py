"""WSGI entrypoint for production servers."""

from app import app

if __name__ == "__main__":
    app.run()
