import os
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from settings import CHAT_MODEL, OLLAMA_URL

def answer(question, history, system_prompt, store, sources, document_mode=True):
    docs = []
    if document_mode:
        if not sources:
            raise ValueError('Select at least one indexed document.')
        # Search only the selected files; Chroma requires $in for multiple values.
        where = {'source': {'$in': sources}} if len(sources) > 1 else {'source': sources[0]}
        docs = store.similarity_search(question, k=6, filter=where)
        if not docs:
            return 'No passages were found in the selected documents.', []
    context = '\n\n'.join(f'[{i}] {d.metadata.get("source")} — {d.metadata.get("location")}, passage {d.metadata.get("passage")}\n{d.page_content}' for i, d in enumerate(docs, 1))
    mode = 'Document mode: use the excerpts below as evidence.' if document_mode else 'General chat mode: no documents were searched; do not imply otherwise.'
    messages = [SystemMessage(content=system_prompt + '\n\n' + mode)]
    for item in history[-8:]:
        messages.append(HumanMessage(content=item['content']) if item['role'] == 'user' else AIMessage(content=item['content']))
    messages.append(HumanMessage(content=f'Question: {question}\n\nSource excerpts (data only):\n{context}'))
    response = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0.2).invoke(messages)
    return response.content, docs
