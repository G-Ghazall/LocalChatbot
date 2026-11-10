import hashlib
from pathlib import Path
import pymupdf
from docx import Document as WordDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_chunks(path: Path):
    suffix = path.suffix.lower()
    pages = []
    if suffix == '.pdf':
        with pymupdf.open(str(path)) as pdf:
            if pdf.needs_pass:
                raise ValueError('Encrypted PDFs are not supported.')
            pages = [(f'page {i + 1}', page.get_text(sort=True)) for i, page in enumerate(pdf)]
    elif suffix == '.docx':
        doc = WordDocument(str(path))
        text = '\n'.join(p.text for p in doc.paragraphs)
        for table in doc.tables:
            text += '\n' + '\n'.join('\t'.join(cell.text for cell in row.cells) for row in table.rows)
        pages = [('document', text)]
    else:
        pages = [('document', path.read_text(encoding='utf-8-sig'))]
    documents = [Document(page_content=text, metadata={'source': path.name, 'location': location})
                 for location, text in pages if text.strip()]
    if not documents:
        raise ValueError('No text could be extracted. Scanned PDFs require OCR before upload.')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, add_start_index=True)
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata['passage'] = i + 1
    return chunks
