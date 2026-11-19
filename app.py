from pathlib import Path

import streamlit as st

import settings
from agent import ANSWER_TAG, SKILL_LABELS, build_graph
from chat_history import load_chats, new_chat, save_chats
from index_manager import open_store, sync_documents

DEFAULT_SYSTEM_PROMPT = ("You are a careful research assistant. Answer in the user's language unless asked otherwise. "
                         'Distinguish quotations from your interpretation. Do not claim to have reviewed whole documents '
                         'when only excerpts were provided. Never invent quotations or references.')
SYSTEM_PROMPT_FILE = settings.STORAGE_DIR / 'system_prompt.txt'
TASK_OPTIONS = {'Auto': '', 'Answer a question': 'answer_question', 'Summarize a topic': 'summarize_topic',
                'Summarize selected documents': 'summarize_document', 'Compare information': 'compare_information',
                'Find information': 'find_information'}

st.set_page_config(page_title='Research assistant', page_icon='📚', layout='wide')
ss = st.session_state

# ---------------------------------------------------------------- session state
if 'chats' not in ss:
    try:
        ss.chats = load_chats()
        ss.system_prompt = (SYSTEM_PROMPT_FILE.read_text(encoding='utf-8')
                            if SYSTEM_PROMPT_FILE.exists() else DEFAULT_SYSTEM_PROMPT)
    except (OSError, ValueError) as exc:
        st.error(f'Could not load saved data from {settings.STORAGE_DIR}: {exc}')
        st.stop()
    ss.active_chat, ss.sync_needed, ss.uploader_key = None, True, 0


@st.cache_resource
def get_store():
    return open_store()


@st.cache_resource
def get_graph(_store):
    return build_graph(_store)


def sync(store):
    with st.status('Checking documents…', expanded=True) as status:
        bar, line = status.progress(0.0), status.empty()

        def report(number, total, name, stage, done, count):
            finished = number if stage in ('unchanged', 'complete', 'error', 'finished') else number - 1
            bar.progress(finished / total if total else 1.0)
            if stage == 'extracting':
                line.write(f'Reading {number}/{total}: {name}')
            elif stage == 'embedding':
                line.write(f'Indexing {number}/{total}: {name} — passage {done}/{count}')

        ss.manifest, ss.notices = sync_documents(store, report)
        status.update(label=f'{len(ss.manifest)} document(s) ready', state='complete', expanded=False)
    ss.sync_needed = False


def save_uploads(uploads):
    problems = []
    for upload in uploads:
        name = upload.name or ''
        if Path(name).name != name or name in {'', '.', '..'} or Path(name).suffix.lower() not in settings.SUPPORTED:
            problems.append(f'Invalid file name: {name}')
        elif upload.size > settings.MAX_FILE_BYTES:
            problems.append(f'{name} exceeds the 50 MB limit.')
        elif (settings.DATA_DIR / name).exists():
            problems.append(f'{name} already exists. Remove or rename it first.')
        else:
            (settings.DATA_DIR / name).write_bytes(upload.getbuffer())
    return problems


def show_message(message, debug):
    with st.chat_message(message['role']):
        if message['role'] == 'assistant' and message.get('skill'):
            st.caption(SKILL_LABELS.get(message['skill'], ''))
        st.markdown(message['content'])
        for warning in message.get('warnings', []):
            st.warning(warning, icon='⚠️')
        for source in message.get('sources', []):
            with st.expander(source['label']):
                st.text(source['text'])
        if debug and message.get('diagnostics'):
            with st.expander('Developer details'):
                st.json(message['diagnostics'])


def run_agent(store, inputs):
    """Stream the answer into the page and return the final graph state."""
    status, body = st.empty(), st.empty()
    text, final = '', {}
    for kind, data in get_graph(store).stream(inputs, stream_mode=['custom', 'messages', 'values']):
        if kind == 'custom':
            status.caption(data.get('progress', ''))
        elif kind == 'messages':
            chunk, meta = data
            if ANSWER_TAG in meta.get('tags', []) and isinstance(chunk.content, str):
                text += chunk.content
                body.markdown(text + '▌')
        else:
            final = data
    return final


def source_cards(result):
    docs, cited = result.get('sources', []), result.get('cited', [])
    numbers = cited or range(1, len(docs) + 1)
    if len(numbers) > 30:  # e.g. an uncited whole-document summary
        numbers = list(numbers)[:30]
    return [{'label': f'[{n}] {docs[n - 1].metadata.get("source")} — {docs[n - 1].metadata.get("location")}',
             'text': docs[n - 1].page_content} for n in numbers]


# ---------------------------------------------------------------- index
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
st.title('Research assistant')
st.caption('Your documents, search index and models run locally on this computer.')
try:
    store = get_store()
    if ss.sync_needed:
        sync(store)
except Exception as exc:
    st.error(f'Could not open or update the document index: {exc}')
    st.info(f'Check that Ollama is running and that `{settings.EMBED_MODEL}` is installed (`ollama list`).')
    if st.button('Try again'):
        st.rerun()
    st.stop()
names = sorted(ss.get('manifest', {}))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header('Documents')
    uploads = st.file_uploader('Add documents', type=[s[1:] for s in settings.SUPPORTED],
                               accept_multiple_files=True, key=f'uploader_{ss.uploader_key}')
    if uploads and st.button('Save and index', type='primary', width='stretch'):
        ss.upload_errors = save_uploads(uploads)
        ss.uploader_key += 1
        ss.sync_needed = True
        st.rerun()
    for error in ss.pop('upload_errors', []):
        st.warning(error)
    if st.button('Refresh documents', width='stretch', help='Re-check the Data folder for changes'):
        ss.sync_needed = True
        st.rerun()
    for notice in ss.get('notices', []):
        (st.warning if notice.startswith('Could not') else st.caption)(notice)
    st.caption(f'{len(names)} indexed document(s)')
    if names:
        with st.expander('Remove a document'):
            target = st.selectbox('Document', names, index=None, placeholder='Choose a document')
            if st.button('Delete file and its index', disabled=not target, width='stretch'):
                path = settings.DATA_DIR / target
                if path.is_file() and not path.is_symlink():
                    path.unlink()
                ss.sync_needed = True
                st.rerun()

    st.divider()
    st.header('Chats')
    if st.button('New chat', width='stretch'):
        ss.active_chat = None
        st.rerun()
    for chat in reversed(ss.chats):
        left, right = st.columns([5, 1])
        if left.button(chat['title'], key=f"open_{chat['id']}", width='stretch',
                       type='primary' if chat['id'] == ss.active_chat else 'secondary'):
            ss.active_chat = chat['id']
            st.rerun()
        if right.button('🗑️', key=f"delete_{chat['id']}", help='Delete chat'):
            ss.delete_chat = chat['id']
            st.rerun()
        if ss.get('delete_chat') == chat['id']:
            yes, no = st.columns(2)
            if yes.button('Delete', key=f"yes_{chat['id']}", type='primary', width='stretch'):
                remaining = [c for c in ss.chats if c['id'] != chat['id']]
                try:
                    save_chats(remaining)
                except OSError as exc:
                    st.error(f'Could not delete chat: {exc}')
                else:
                    ss.chats, ss.delete_chat = remaining, None
                    ss.active_chat = None if ss.active_chat == chat['id'] else ss.active_chat
                    st.rerun()
            if no.button('Cancel', key=f"no_{chat['id']}", width='stretch'):
                ss.delete_chat = None
                st.rerun()
    st.divider()
    st.caption(f'Chat model: {settings.CHAT_MODEL} · Embeddings: {settings.EMBED_MODEL}')

# ---------------------------------------------------------------- main
mode = st.radio('Mode', ['Search documents', 'General chat'], horizontal=True, label_visibility='collapsed')
documents = mode == 'Search documents'
selected = st.multiselect('Documents to search', names, default=names, disabled=not documents)
with st.expander('Options'):
    col1, col2 = st.columns(2)
    task = TASK_OPTIONS[col1.selectbox('Task', list(TASK_OPTIONS), help='Auto picks a task from your wording.')]
    detail = col2.selectbox('Summary length', ['Brief', 'Standard', 'Detailed'], index=1)
    verify = st.checkbox('Double-check answers against cited sources (slower)', value=settings.VERIFY)
    debug = st.checkbox('Show developer details', value=settings.DEBUG)
with st.expander('System prompt'):
    if st.button('Restore default'):
        ss.system_prompt = DEFAULT_SYSTEM_PROMPT
    system_prompt = st.text_area('Instructions for the assistant', key='system_prompt', height=150, max_chars=10000)
    if st.button('Save system prompt'):
        try:
            settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            SYSTEM_PROMPT_FILE.write_text(system_prompt, encoding='utf-8')
            st.success('Saved.')
        except OSError as exc:
            st.error(f'Could not save the system prompt: {exc}')

chat = next((c for c in ss.chats if c['id'] == ss.active_chat), None)
history = chat['messages'] if chat else []
for message in history:
    show_message(message, debug)

question = st.chat_input('Ask a question in Swedish or English')
if question:
    st.chat_message('user').markdown(question)
    with st.chat_message('assistant'):
        try:
            result = run_agent(store, {
                'question': question, 'mode': 'document' if documents else 'general',
                'documents': selected if documents else [], 'task': task if documents else '',
                'detail': detail, 'verify': verify, 'system_prompt': system_prompt,
                'history': [{'role': m['role'], 'content': m['content']} for m in history]})
        except Exception as exc:
            st.error(f'Could not answer: {exc}')
            st.info(f'Check that Ollama is running and `{settings.CHAT_MODEL}` is installed (`ollama list`).')
            st.stop()
    reply = {'role': 'assistant', 'content': result['answer'], 'skill': result.get('skill'),
             'warnings': result.get('warnings', []), 'sources': source_cards(result),
             'selected_documents': selected if documents else [], 'model': settings.CHAT_MODEL,
             'diagnostics': {'skill': result.get('skill'), 'search_query': result.get('query'),
                             'comparison_items': result.get('items'), 'cited': result.get('cited'),
                             'retrieved_passages': len(result.get('sources', []))}}
    chats = [dict(c, messages=list(c['messages'])) for c in ss.chats]
    current = next((c for c in chats if c['id'] == ss.active_chat), None)
    if current is None:
        current = new_chat(question)
        chats.append(current)
    current['messages'] += [{'role': 'user', 'content': question}, reply]
    try:
        save_chats(chats)
    except OSError as exc:
        st.error(f'Could not save the chat: {exc}')
    ss.chats, ss.active_chat = chats, current['id']
    st.rerun()
