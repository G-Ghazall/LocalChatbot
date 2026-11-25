from pathlib import Path
import os


def _env(name, default, cast=str):
    return cast(os.getenv(f'LOCALCHAT_{name}', default))


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'Data'
STORAGE_DIR = ROOT / 'storage'

OLLAMA_URL = _env('OLLAMA_URL', 'http://127.0.0.1:11434')
CHAT_MODEL = _env('CHAT_MODEL', 'llama3.2:latest')
EMBED_MODEL = _env('EMBED_MODEL', 'nomic-embed-text')
NUM_CTX = _env('NUM_CTX', '8192', int)             # model context window in tokens (Ollama default is too small)
MAX_CONTEXT = _env('MAX_CONTEXT', '16000', int)    # max characters of source text per model call

FOCUSED_K = _env('FOCUSED_K', '6', int)            # passages for questions / find
TOPIC_K = _env('TOPIC_K', '12', int)               # passages for topic summaries
COMPARISON_K = _env('COMPARISON_K', '5', int)      # passages per compared item
MAX_SUMMARY_PARTS = _env('MAX_SUMMARY_PARTS', '30', int)  # cap on model calls for whole-document summaries
HISTORY_TURNS = _env('HISTORY_TURNS', '6', int)    # previous messages sent to the model

VERIFY = _env('VERIFY', 'false').lower() == 'true'  # default for the "double-check answers" option
DEBUG = _env('DEBUG', 'false').lower() == 'true'

CHUNK_SIZE = _env('CHUNK_SIZE', '1200', int)
CHUNK_OVERLAP = _env('CHUNK_OVERLAP', '200', int)
SUPPORTED = {'.pdf', '.docx', '.txt', '.md'}
MAX_FILE_BYTES = 50 * 1024 * 1024
