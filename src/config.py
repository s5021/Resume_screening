"""
Configuration file for the project
"""
import os
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resume_screening.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")
# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Model Settings
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EMBEDDING_DIM = 384
# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "4"))
# File Upload Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
# Processing Settings
MIN_RESUME_LENGTH = 100
MAX_RESUME_LENGTH = 50000
# Scoring Weights
WEIGHTS = {
    "semantic_similarity": 0.5,
    "skill_match": 0.3,
    "experience_match": 0.2
}
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
