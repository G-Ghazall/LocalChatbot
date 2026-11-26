import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import settings
from agent import ANSWER_TAG, NO_DOCUMENTS, NO_PASSAGES, build_graph, citations, comparison_items, fit, pack, route
from chat_history import load_chats, save_chats


def doc(text, passage, source='a.txt', page=1):
    return Document(page_content=text, metadata={'source': source, 'passage': passage, 'location': f'page {page}'})


class Store:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [doc('Deadline is 15 January.', 2), doc('First section.', 1)]
        self.calls = []

    def similarity_search(self, query, k, filter):
        self.calls.append(('similarity', query, k, filter))
        return self.docs[:k]

    def max_marginal_relevance_search(self, query, k, fetch_k, filter):
        self.calls.append(('mmr', query, k, filter))
        return self.docs[:k]

    def get(self, where, include):
        docs = [d for d in self.docs if d.metadata['source'] == where['source']]
        return {'documents': [d.page_content for d in docs], 'metadatas': [d.metadata for d in docs]}


def model(*responses):
    return FakeListChatModel(responses=list(responses))


def run(store, answers, checker=(), **inputs):
    state = dict(question='What is the deadline?', mode='document', documents=['a.txt'], history=[], system_prompt='')
    state.update(inputs)
    return build_graph(store, model(*answers), model(*checker) if checker else model('{}')).invoke(state)


@pytest.mark.parametrize('message,expected', [
    ('What is the deadline?', 'answer_question'),
    ('Which AI job profiles were most in demand?', 'answer_question'),
    ('Summarize the admission rules', 'summarize_topic'),
    ('summarize all info about AI in documents', 'summarize_topic'),
    ('Summarize what the reports say about AI', 'summarize_topic'),
    ('Summarize this document', 'summarize_document'),
    ('Sammanfatta rapporten', 'summarize_topic'),
    ('Sammanfatta hela rapporten', 'summarize_document'),
    ('Compare A and B', 'compare_information'),
    ('Jämför Sverige och Finland', 'compare_information'),
    ('Compare the requirements', 'compare_information'),
    ('Where is the deadline?', 'find_information'),
    ('Var nämns AI-kompetens?', 'find_information'),
])
def test_route(message, expected):
    assert route(message, 'document', ['a.txt'])[0] == expected


def test_route_modes_and_override():
    assert route('Summarize this document', 'general', [])[0] == 'general_chat'
    assert route('hello', 'general', [], 'find_information')[0] == 'general_chat'
    assert route('hello', 'document', [])[0] == 'clarify_question'
    assert route('hello', 'document', ['a'], 'find_information')[0] == 'find_information'
    assert comparison_items('Compare Program A and Program B?') == ['Program A', 'Program B']
    assert comparison_items('What is the difference between Sweden and Finland') == ['Sweden', 'Finland']


def test_citations_and_packing():
    assert citations('A [1], B [2][3] and C [4, 5].') == {1, 2, 3, 4, 5}
    assert citations('No refs') == set()
    docs = [doc('x' * 10, i) for i in range(5)]
    assert len(fit(docs, 25)) == 2
    assert len(fit([doc('x' * 100, 1)], 10)) == 1  # always keep at least one passage
    assert pack([1, 1, 2, 2], 100, size=lambda x: 1, group=lambda x: x) == [[1, 1], [2, 2]]


def test_answer_with_valid_citation():
    store = Store()
    result = run(store, ['Deadline is 15 January [1].'])
    assert result['answer'] == 'Deadline is 15 January [1].'
    assert result['cited'] == [1] and result['warnings'] == []
    assert store.calls[0][3] == {'source': 'a.txt'}


def test_multiple_documents_use_in_filter():
    store = Store()
    run(store, ['Answer [1].'], documents=['a.txt', 'b.txt'])
    assert store.calls[0][3] == {'source': {'$in': ['a.txt', 'b.txt']}}


def test_bad_citations_are_flagged_not_hidden():
    result = run(Store(), ['Invented [9].'])
    assert result['answer'] == 'Invented [9].'
    assert any('[9]' in w for w in result['warnings'])
    result = run(Store(), ['No citations here.'])
    assert result['answer'] == 'No citations here.'
    assert any('no valid source citations' in w for w in result['warnings'])


def test_no_passages():
    result = run(Store([]), ['unused'])
    assert result['answer'] == NO_PASSAGES and result['sources'] == []


def test_no_documents_selected_asks_for_selection():
    store = Store()
    result = run(store, [], documents=[])
    assert result['answer'] == NO_DOCUMENTS and not store.calls


def test_general_chat_skips_retrieval():
    store = Store()
    result = run(store, ['Hello!'], mode='general', documents=[], task='answer_question')
    assert result['answer'] == 'Hello!' and not store.calls and result['skill'] == 'general_chat'


def test_follow_up_is_rewritten_for_search():
    store = Store()
    history = [{'role': 'user', 'content': 'Tell me about program A'}, {'role': 'assistant', 'content': 'It is ...'}]
    result = run(store, ['Program A deadline', 'Deadline is 15 January [1].'], history=history)
    assert store.calls[0][1] == 'Program A deadline'
    assert result['answer'] == 'Deadline is 15 January [1].'


def test_topic_summary_uses_diverse_search():
    store = Store()
    run(store, ['Summary [1].'], question='Summarize the admission rules')
    assert store.calls[0][0] == 'mmr'


def test_compare_searches_each_item():
    store = Store()
    result = run(store, ['A [1] vs B [2].'], question='Compare program A and program B')
    assert [c[1] for c in store.calls] == ['Compare program A and program B program A',
                                           'Compare program A and program B program B']
    assert result['items'] == ['program A', 'program B']


def test_whole_document_summary_map_reduce(monkeypatch):
    monkeypatch.setattr(settings, 'MAX_CONTEXT', 30)
    store = Store([doc('First part text.', 1), doc('Second part text.', 2), doc('Third part text.', 3)])
    result = run(store, ['note 1 [1]', 'note 2 [2]', 'note 3 [3]', 'Summary [1][3].'], question='Summarize this document')
    assert result['skill'] == 'summarize_document'
    assert result['answer'] == 'Summary [1][3].' and result['cited'] == [1, 3]
    assert [d.metadata['passage'] for d in result['sources']] == [1, 2, 3]


def test_too_large_document_falls_back_to_topic_summary(monkeypatch):
    monkeypatch.setattr(settings, 'MAX_CONTEXT', 10)
    monkeypatch.setattr(settings, 'MAX_SUMMARY_PARTS', 1)
    store = Store([doc('First part text.', 1), doc('Second part text.', 2)])
    result = run(store, ['Summary [1].'], question='Summarize this document')
    assert result['answer'] == 'Summary [1].' and 'too long' in result['warnings'][0]


def test_verify_adds_warning_but_keeps_answer():
    rejected = '{"supported": false, "unsupported_claims": ["Deadline is 1 Feb"]}'
    result = run(Store(), ['Deadline is 1 Feb [1].'], checker=[rejected], verify=True)
    assert result['answer'] == 'Deadline is 1 Feb [1].'
    assert any('Deadline is 1 Feb' in w for w in result['warnings'])
    result = run(Store(), ['Deadline is 15 January [1].'], checker=['{"supported": true}'], verify=True)
    assert result['warnings'] == []
    result = run(Store(), ['Deadline [1].'], checker=['not json'], verify=True)
    assert any('could not be completed' in w for w in result['warnings'])


def test_answer_tokens_are_streamed():
    graph = build_graph(Store(), model('Deadline is 15 January [1].'), model('{}'))
    state = dict(question='What is the deadline?', mode='document', documents=['a.txt'], history=[])
    tokens = [chunk.content for chunk, meta in graph.stream(state, stream_mode='messages')
              if ANSWER_TAG in meta.get('tags', [])]
    assert ''.join(tokens) == 'Deadline is 15 January [1].'


def test_history_round_trip(tmp_path, monkeypatch):
    import chat_history
    monkeypatch.setattr(chat_history, 'STORAGE_DIR', tmp_path)
    monkeypatch.setattr(chat_history, 'HISTORY_FILE', tmp_path / 'chats.json')
    chats = [{'id': '1', 'title': 't', 'messages': [{'role': 'assistant', 'content': 'old'}]}]
    save_chats(chats)
    assert load_chats() == chats
