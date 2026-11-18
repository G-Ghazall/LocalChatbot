from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Data'
STORAGE_DIR = ROOT / 'storage'
CHAT_MODEL = os.getenv('LOCALCHAT_CHAT_MODEL', 'llama3.2:latest')
EMBED_MODEL = os.getenv('LOCALCHAT_EMBED_MODEL', 'nomic-embed-text')
OLLAMA_URL = os.getenv('LOCALCHAT_OLLAMA_URL', 'http://127.0.0.1:11434')
PLANNER_MODEL = os.getenv('LOCALCHAT_PLANNER_MODEL', CHAT_MODEL)
FOCUSED_K = int(os.getenv('LOCALCHAT_FOCUSED_K', '6'))
TOPIC_K = int(os.getenv('LOCALCHAT_TOPIC_K', '12'))
COMPARISON_K = int(os.getenv('LOCALCHAT_COMPARISON_K', '6'))
TOPIC_QUERIES = int(os.getenv('LOCALCHAT_TOPIC_QUERIES', '4'))
MAX_PER_DOCUMENT = int(os.getenv('LOCALCHAT_MAX_PER_DOCUMENT', '12'))
MAX_CONTEXT = int(os.getenv('LOCALCHAT_MAX_CONTEXT', '24000'))
CONFIDENCE_THRESHOLD = float(os.getenv('LOCALCHAT_CONFIDENCE_THRESHOLD', '0.55'))
DEBUG = os.getenv('LOCALCHAT_DEBUG', 'false').lower() == 'true'
CHUNK_SIZE = int(os.getenv('LOCALCHAT_CHUNK_SIZE', '1200'))
CHUNK_OVERLAP = int(os.getenv('LOCALCHAT_CHUNK_OVERLAP', '200'))
SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}
MAX_FILE_BYTES = 50 * 1024 * 1024
