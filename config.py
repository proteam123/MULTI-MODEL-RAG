import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

PDF_DIR = Path("data")
IMAGE_DIR = Path("data/extracted_images")
PDF_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-20b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "novacore-multimodal-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "fy2026-demo")
PINECONE_HISTORY_NAMESPACE = f"{PINECONE_NAMESPACE}-history"
