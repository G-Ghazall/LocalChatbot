import hashlib
import json
import os
import logging
from datetime import datetime, timezone
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from document_processing import digest, load_chunks
from settings import DATA_DIR, STORAGE_DIR, EMBED_MODEL, OLLAMA_URL, SUPPORTED, MAX_FILE_BYTES


def open_store():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    # A separate collection prevents mixing vectors from different embedding models.
    collection = 'docs_' + hashlib.sha256(EMBED_MODEL.encode()).hexdigest()[:16]
    return Chroma(collection_name=collection, persist_directory=str(STORAGE_DIR / 'chroma'),
                  embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL))


def sync_documents(store, progress=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(EMBED_MODEL.encode()).hexdigest()[:16]
    manifest_path = STORAGE_DIR / f'manifest_{key}.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            raise ValueError('Invalid index manifest')
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'Cannot read index manifest: {exc}. Do not delete it without rebuilding the index.') from exc
    # Reconcile interrupted writes: only manifest-owned IDs are eligible for retrieval.
    stored_ids = set(store.get(include=[])['ids'])
    known_ids = {item for entry in manifest.values() for item in entry['ids']}
    orphan_ids = list(stored_ids - known_ids)
    if orphan_ids:
        store.delete(ids=orphan_ids)
        stored_ids.difference_update(orphan_ids)
    files = {p.name: p for p in DATA_DIR.iterdir() if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED}
    messages = []
    for name in sorted(set(manifest) - set(files)):
        store.delete(ids=manifest[name]['ids'])
        del manifest[name]
        messages.append(f'Removed {name} from the index.')
    total = len(files)
    for number, (name, path) in enumerate(sorted(files.items()), start=1):
        def report(stage, done=0, count=0):
            if progress:
                progress(number, total, name, stage, done, count)
        report('checking')
        ids = []
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError('File exceeds the 50 MB limit.')
            file_hash = digest(path)
            entry = manifest.get(name, {})
            parser_version = 'pymupdf-v1' if path.suffix.lower() == '.pdf' else 'v1'
            if (entry.get('hash') == file_hash and entry.get('parser_version') == parser_version
                    and set(entry.get('ids', [])) <= stored_ids):
                report('unchanged')
                continue
            report('extracting')
            chunks = load_chunks(path)
            # Include the content hash so a failed update leaves the previous index intact.
            ids = [hashlib.sha256(f'{name}\0{file_hash}\0{parser_version}\0{i}'.encode()).hexdigest() for i in range(len(chunks))]
            indexed_at = datetime.now(timezone.utc).isoformat()
            for chunk, chunk_id in zip(chunks, ids):
                chunk.metadata.update({'chunk_id': chunk_id, 'document_id': hashlib.sha256(name.encode()).hexdigest(),
                                       'file_hash': file_hash, 'file_type': path.suffix.lower(), 'indexed_at': indexed_at})
            for start in range(0, len(chunks), 32):
                report('embedding', start, len(chunks))
                store.add_documents(chunks[start:start + 32], ids=ids[start:start + 32])
            report('embedding', len(chunks), len(chunks))
            old_ids = manifest.get(name, {}).get('ids', [])
            if old_ids:
                store.delete(ids=old_ids)
            manifest[name] = {'hash': file_hash, 'ids': ids, 'parser_version': parser_version,
                              'size': path.stat().st_size, 'indexed_at': indexed_at, 'chunk_count': len(chunks)}
            messages.append(f'Indexed {name}: {len(chunks)} passages.')
            report('complete')
        except Exception as exc:
            logging.getLogger(__name__).warning('Indexing failed for %s: %s', name, type(exc).__name__)
            # Never answer from an outdated version when its replacement failed.
            if ids:
                store.delete(ids=ids)
            if name in manifest:
                store.delete(ids=manifest[name]['ids'])
                del manifest[name]
            messages.append(f'Could not index {name}: {exc}')
            report('error')
        temp = manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
        os.replace(temp, manifest_path)
    temp = manifest_path.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, manifest_path)
    if progress:
        progress(total, total, '', 'finished', 0, 0)
    return manifest, messages
