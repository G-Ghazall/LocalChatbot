import json
import os
import tempfile
from uuid import uuid4

from settings import STORAGE_DIR

HISTORY_FILE = STORAGE_DIR / 'chats.json'


def load_chats():
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open(encoding='utf-8') as file:
        chats = json.load(file)
    if not isinstance(chats, list):
        raise ValueError('Chat history has an invalid format')
    return chats


def save_chats(chats):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=STORAGE_DIR,
                                         delete=False, suffix='.tmp') as file:
            temporary = file.name
            json.dump(chats, file, ensure_ascii=False, indent=2)
        os.replace(temporary, HISTORY_FILE)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def new_chat(question):
    title = ' '.join(question.split())
    return {'id': uuid4().hex, 'title': title[:50] + ('…' if len(title) > 50 else ''), 'messages': []}
