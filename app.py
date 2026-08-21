"""
Root entry point for deployment platforms (Render, Heroku, Railway, Vercel).
Re-exports the FastAPI `app` instance from backend.app.
"""
import os
import sys

# Add backend directory to sys.path so internal imports resolve
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.app import app

__all__ = ["app"]
