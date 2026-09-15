import os

from dotenv import load_dotenv

load_dotenv()

# Override with GEMINI_MODEL in .env if needed.
# gemini-2.5-* is blocked for new API keys; gemini-3.6-flash has a tight free RPD.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
