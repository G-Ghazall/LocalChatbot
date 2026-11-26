"""Keeps the Chroma index in sync with the files in Data/ (only changed files are re-embedded)."""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from document_processing import digest, load_chunks
from settings import DATA_DIR, EMBED_MODEL, MAX_FILE_BYTES, OLLAMA_URL, STORAGE_DIR, SUPPORTED

log = logging.getLogger(__name__)
# One collection and manifest per embedding model, so vectors from different models never mix.
MODEL_KEY = hashlib.sha256(EMBED_MODEL.encode()).hexdigest()[:16]
MANIFEST = STORAGE_DIR / f'manifest_{MODEL_KEY}.json'
BATCH = 32


def open_store():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(collection_name=f'docs_{MODEL_KEY}', persist_directory=str(STORAGE_DIR / 'chroma'),
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL))


def _load_manifest():
    try:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {}
        if not isinstance(manifest, dict):
            raise ValueError('invalid format')
        return manifest
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'Cannot read index manifest: {exc}. Do not delete it without rebuilding the index.') from exc


def _save_manifest(manifest):
    temp = MANIFEST.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, MANIFEST)


def _index_file(store, name, path, file_hash, parser_version, report):
    chunks = load_chunks(path)
    # IDs include the content hash so a failed update never overwrites the previous version.
    ids = [hashlib.sha256(f'{name}\0{file_hash}\0{parser_version}\0{i}'.encode()).hexdigest()
           for i in range(len(chunks))]
    indexed_at = datetime.now(timezone.utc).isoformat()
    for chunk, chunk_id in zip(chunks, ids):
        chunk.metadata.update({'chunk_id': chunk_id, 'file_hash': file_hash,
                               'file_type': path.suffix.lower(), 'indexed_at': indexed_at})
    try:
        for start in range(0, len(chunks), BATCH):
            report('embedding', start, len(chunks))
            store.add_documents(chunks[start:start + BATCH], ids=ids[start:start + BATCH])
    except Exception:
        store.delete(ids=ids)
        raise
    report('embedding', len(chunks), len(chunks))
    return {'hash': file_hash, 'ids': ids, 'parser_version': parser_version, 'size': path.stat().st_size,
            'indexed_at': indexed_at, 'chunk_count': len(chunks)}


def sync_documents(store, progress=None):
    """Index new/changed files and drop removed ones. Returns (manifest, notices)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    # Remove vectors left behind by interrupted runs; only manifest-owned IDs are searchable.
    stored = set(store.get(include=[])['ids'])
    orphans = list(stored - {i for entry in manifest.values() for i in entry['ids']})
    if orphans:
        store.delete(ids=orphans)
        stored.difference_update(orphans)

    files = {p.name: p for p in DATA_DIR.iterdir()
             if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED}
    notices = []
    for name in sorted(set(manifest) - set(files)):
        store.delete(ids=manifest.pop(name)['ids'])
        notices.append(f'Removed {name} from the index.')

    total = len(files)
    for number, (name, path) in enumerate(sorted(files.items()), start=1):
        def report(stage, done=0, count=0):
            if progress:
                progress(number, total, name, stage, done, count)
        report('checking')
        old = manifest.get(name, {})
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('File exceeds the 50 MB limit.')
            file_hash = digest(path)
            parser_version = 'pymupdf-v1' if path.suffix.lower() == '.pdf' else 'v1'
            if (old.get('hash') == file_hash and old.get('parser_version') == parser_version
                    and set(old.get('ids', [])) <= stored):
                report('unchanged')
                continue
            report('extracting')
            manifest[name] = _index_file(store, name, path, file_hash, parser_version, report)
            stale = set(old.get('ids', [])) - set(manifest[name]['ids'])
            if stale:
                store.delete(ids=list(stale))
            notices.append(f'Indexed {name}: {manifest[name]["chunk_count"]} passages.')
            report('complete')
        except Exception as exc:
            log.warning('Indexing failed for %s: %s', name, type(exc).__name__)
            # Never answer from an outdated version when its replacement failed.
            if name in manifest:
                store.delete(ids=manifest.pop(name)['ids'])
            notices.append(f'Could not index {name}: {exc}')
            report('error')
        _save_manifest(manifest)
    _save_manifest(manifest)
    if progress:
        progress(total, total, '', 'finished', 0, 0)
    return manifest, notices
