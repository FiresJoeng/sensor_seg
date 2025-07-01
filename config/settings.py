# new_sensor_project/config/settings.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- Base Directory ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Environment Variables ---
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

# --- API Configuration ---
# OpenAI-Compatible API (for legacy methods or other file types)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = OPENAI_API_KEY if OPENAI_API_KEY else os.getenv("LLM_API_KEY") # Fallback
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "default-model-name")
LLM_API_URL = os.getenv("LLM_API_URL", "default-api-url")

# Gemini API Configuration (for native PDF processing)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash") # Use a specific, capable model

# --- NEW: PDF Processing Method ---
# Choose how to process PDF files:
# 'legacy': Use the old method (pdfplumber + camelot + image encoding).
# 'gemini': Use the new method with Gemini API to process PDF directly.
PDF_PROCESSING_METHOD = os.getenv("PDF_PROCESSING_METHOD", "gemini").lower()


# LLM General Settings
LLM_TEMPERATURE = 0.1
LLM_REQUEST_TIMEOUT = 300

# --- Warnings for missing keys ---
if not LLM_API_KEY:
    print("警告: 未在 .env 文件中找到 OPENAI_API_KEY 或 LLM_API_KEY。处理非PDF文件或使用 'legacy' PDF 方法时可能会失败。")
if PDF_PROCESSING_METHOD == 'gemini' and not GEMINI_API_KEY:
    print("警告: PDF_PROCESSING_METHOD 设置为 'gemini'，但未在 .env 文件中找到 GEMINI_API_KEY。PDF处理将失败。")


# --- File Paths ---
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
KB_SOURCE_DIR = KNOWLEDGE_BASE_DIR / "source"
KB_VECTOR_STORE_DIR = KNOWLEDGE_BASE_DIR / "vector_store"
VECTOR_STORE_PATH = KB_VECTOR_STORE_DIR
LIBS_DIR = BASE_DIR / "libs"
STANDARD_LIBS_DIR = LIBS_DIR / "standard"
STANDARD_INDEX_JSON = STANDARD_LIBS_DIR / "transmitter" / "index.json"

# --- Logging Configuration ---
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_TO_FILE = True
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# --- Standard Matching Configuration ---
FUZZY_MATCH_THRESHOLD = 0.7
MAIN_MODEL_SIMILARITY_THRESHOLD = 0.8

# --- Ensure Directories Exist ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KB_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Initial Print Statement ---
print(f"--- Configuration Loaded ---")
print(f"Project Base Directory: {BASE_DIR}")
print(f"Log Level: {LOG_LEVEL_STR}")
print(f"PDF Processing Method: {PDF_PROCESSING_METHOD.upper()}")
print(f"Gemini API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
print(f"Gemini Model Name: {GEMINI_MODEL_NAME}")
print(f"--- End Configuration ---")
