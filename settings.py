from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Data'
STORAGE_DIR = ROOT / 'storage'
CHAT_MODEL = os.getenv('LOCALCHAT_CHAT_MODEL', 'llama3.2:latest')
EMBED_MODEL = os.getenv('LOCALCHAT_EMBED_MODEL', 'nomic-embed-text')
OLLAMA_URL = 'http://127.0.0.1:11434'
SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}
MAX_FILE_BYTES = 50 * 1024 * 1024
