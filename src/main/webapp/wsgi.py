"""WSGI entrypoint for production servers."""

from src.main.webapp.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
