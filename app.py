from pathlib import Path
import streamlit as st
from settings import DATA_DIR, STORAGE_DIR, CHAT_MODEL, EMBED_MODEL, SUPPORTED, MAX_FILE_BYTES
from index_manager import open_store, sync_documents
from rag import answer

st.set_page_config(page_title='Research assistant', page_icon='📚', layout='wide')
DATA_DIR.mkdir(parents=True, exist_ok=True)
st.title('Research assistant')
st.caption('Your documents, search index and model run locally on this computer.')

if 'history' not in st.session_state:
    st.session_state.history = []
if 'sync_needed' not in st.session_state:
    st.session_state.sync_needed = True

@st.cache_resource
def get_store():
    return open_store()

try:
    store = get_store()
    if st.session_state.sync_needed:
        status = st.status('Checking documents and updating the local index...', expanded=True)
        bar = status.progress(0)
        detail = status.empty()
        def show_progress(number, total, filename, stage, done, count):
            completed = number if stage in ('unchanged', 'complete', 'error', 'finished') else number - 1
            bar.progress(completed / total if total else 1.0)
            if stage == 'finished':
                detail.write(f'Finished checking {total} document(s).')
                return
            labels = {
                'checking': 'Checking', 'extracting': 'Extracting text from',
                'embedding': 'Indexing', 'unchanged': 'Already up to date:',
                'complete': 'Indexed', 'error': 'Could not index',
            }
            extra = f' - passages {done}/{count}' if stage == 'embedding' else ''
            detail.write(f'{labels[stage]} document {number} of {total}: {filename}{extra}. {total - completed} file(s) remaining.')
        manifest, notices = sync_documents(store, progress=show_progress)
        status.update(label='Document check complete', state='complete', expanded=False)
        st.session_state.manifest = manifest
        st.session_state.notices = notices
        st.session_state.sync_needed = False
except Exception as exc:
    st.error(f'Could not initialize or synchronize the index: {exc}')
    st.info(f'Check that Ollama is running and the embedding model {EMBED_MODEL} is installed. See the README.')
    if st.button('Try again'):
        st.session_state.sync_needed = True
        st.rerun()
    st.stop()

with st.sidebar:
    st.header('Document library')
    st.caption('Files in the Data folder are checked on startup and when you refresh.')
    uploads = st.file_uploader('Add documents', type=['pdf', 'docx', 'txt', 'md'], accept_multiple_files=True)
    if st.button('Save uploaded files', disabled=not uploads, use_container_width=True):
        problems = []
        for upload in uploads or []:
            name = upload.name
            if not name or name in {'.', '..'} or '/' in name or '\\' in name or Path(name).name != name or Path(name).suffix.lower() not in SUPPORTED:
                problems.append(f'Invalid filename: {name}')
                continue
            if upload.size > MAX_FILE_BYTES:
                problems.append(f'{name} exceeds the 50 MB limit.')
                continue
            target = DATA_DIR / name
            if target.exists() or target.is_symlink():
                problems.append(f'{name} already exists. Remove or rename it first.')
                continue
            with target.open('xb') as file:
                file.write(upload.getbuffer())
        if problems:
            st.session_state.upload_errors = problems
        st.session_state.sync_needed = True
        st.rerun()
    for error in st.session_state.pop('upload_errors', []):
        st.warning(error)
    if st.button('Refresh documents', use_container_width=True):
        st.session_state.sync_needed = True
        st.rerun()
    for notice in st.session_state.get('notices', []):
        if notice.startswith('Could not'):
            st.warning(notice)
        else:
            st.caption(notice)
    manifest = st.session_state.get('manifest', {})
    names = sorted(manifest)
    st.caption(f'{len(names)} indexed document(s)')
    if names:
        delete_name = st.selectbox('Remove a document', [''] + names, format_func=lambda x: x or 'Choose a document')
        confirm = st.checkbox('I want to delete this file and its indexed passages')
        if st.button('Delete document', disabled=not delete_name or not confirm, use_container_width=True):
            target = DATA_DIR / delete_name
            if target.is_file() and not target.is_symlink():
                target.unlink()
            st.session_state.sync_needed = True
            st.rerun()
    st.divider()
    st.caption(f'Chat model: {CHAT_MODEL}  ·  Embeddings: {EMBED_MODEL}')
    st.caption('Chat is kept in this browser session only. Closing or reloading the session can clear it.')
    if st.button('Clear conversation', use_container_width=True):
        st.session_state.history = []
        st.rerun()

mode = st.radio('Mode', ['Search documents', 'General chat'], horizontal=True)
selected = st.multiselect('Documents to search', names, default=names, disabled=mode != 'Search documents')
with st.expander('Research instructions'):
    instructions = st.text_area('Guide the assistant', value='Distinguish participants’ words from interpretation. Mention uncertainty and differing views.', max_chars=4000)
st.caption('Answers can contain mistakes. Check cited passages before using them in research.')
for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        for source in message.get('sources', []):
            with st.expander(source['label']):
                st.text(source['text'])

question = st.chat_input('Ask a question in Swedish or English')
if question:
    st.chat_message('user').markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Thinking locally…'):
                scope = (mode, tuple(sorted(selected)) if mode == 'Search documents' else ())
                scoped_history = [m for m in st.session_state.history if m.get('scope') == scope]
                response, docs = answer(question, scoped_history, instructions, store, selected, mode == 'Search documents')
            st.markdown(response)
            sources = []
            for i, doc in enumerate(docs, 1):
                label = f'[{i}] {doc.metadata.get("source")} — {doc.metadata.get("location")}, passage {doc.metadata.get("passage")}'
                sources.append({'label': label, 'text': doc.page_content})
                with st.expander(label):
                    st.text(doc.page_content)
            st.session_state.history.extend([{'role': 'user', 'content': question, 'scope': scope}, {'role': 'assistant', 'content': response, 'sources': sources, 'scope': scope}])
        except Exception as exc:
            st.error(f'Could not answer: {exc}')
            st.info('Check the selected documents and confirm that Ollama and the configured models are available.')
