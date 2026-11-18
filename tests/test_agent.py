import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from agent import CHECK_FAILED, NO_PASSAGES, EvidenceChecker, PlannerResult, Retrieval, build_graph, deduplicate, fallback, parse_json
from chat_history import load_chats, save_chats


def plan(skill='answer_question', **changes):
    data = dict(skill=skill, reason='test', rewritten_query='deadline', topics=[], comparison_items=[],
                document_scope=[], retrieval_queries=[], needs_retrieval=True, needs_clarification=False,
                clarification_question=None, confidence=0.9)
    data.update(changes)
    return json.dumps(data)


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.responses.pop(0))


class Store:
    def __init__(self):
        self.calls = []
        self.docs = [Document(page_content='Deadline is 15 January.', metadata={'source': 'a.txt', 'passage': 2, 'location': 'document'}),
                     Document(page_content='First section.', metadata={'source': 'a.txt', 'passage': 1, 'location': 'document'})]

    def similarity_search(self, query, k, filter):
        self.calls.append((query, k, filter))
        return self.docs[:1]

    def get(self, where, include):
        return {'documents': [d.page_content for d in self.docs], 'metadatas': [d.metadata for d in self.docs]}


def checker_response():
    return json.dumps(dict(supported=True, unsupported_claims=[], citation_issues=[], conflicts_found=[], recommended_action='accept', confidence=0.9))


def state(**kwargs):
    result = dict(user_message='What is the deadline?', current_mode='document', selected_document_names=['a.txt'], conversation_context=[], system_prompt='')
    result.update(kwargs)
    return result


@pytest.mark.parametrize('message,expected', [('What is the deadline?', 'answer_question'), ('Summarize the admission rules', 'summarize_topic'), ('summarize all info about AI in documents', 'summarize_topic'), ('Summarize this document', 'summarize_document'), ('Compare A and B', 'compare_information'), ('Where is the deadline?', 'find_information'), ('Compare the requirements', 'clarify_question')])
def test_fallback(message, expected):
    assert fallback(message, 'document', ['a.txt']).skill == expected


def test_general_and_override():
    assert fallback('hello', 'general', []).skill == 'general_chat'
    assert fallback('hello', 'document', ['a'], 'find_information').skill == 'find_information'


def test_json_validation():
    assert parse_json(SimpleNamespace(content='```json\n' + plan() + '\n```'), PlannerResult).skill == 'answer_question'
    for value in (None, '', ' ', 'invalid', plan(skill='unknown'), '{}', plan(confidence=2)):
        with pytest.raises(ValueError):
            parse_json(SimpleNamespace(content=value), PlannerResult)
    with pytest.raises(ValueError):
        parse_json(None, PlannerResult)


def test_repair_and_fallback():
    store = Store()
    planner = Model(None, plan())
    result = build_graph(store, planner, Model('Deadline: 15 January [1]'), Model(checker_response())).invoke(state())
    assert result['retry_count'] == 1 and not result['fallback_used']
    planner = Model(None, 'bad')
    result = build_graph(store, planner, Model('Deadline: 15 January [1]'), Model(checker_response())).invoke(state())
    assert result['fallback_used'] and len(planner.calls) == 2


def test_clarify_and_general_skip_retrieval():
    store = Store()
    p = Model(plan('clarify_question', clarification_question='Which program?'))
    result = build_graph(store, p, Model(), Model()).invoke(state())
    assert result['final_answer'] == 'Which program?' and not store.calls
    p = Model(plan('general_chat'))
    result = build_graph(store, p, Model('Hello!'), Model()).invoke(state(current_mode='general', selected_document_names=[]))
    assert result['final_answer'] == 'Hello!' and not store.calls


def test_real_graph_citation_and_filter():
    store = Store()
    result = build_graph(store, Model(plan()), Model('Deadline is 15 January [1].'), Model(checker_response())).invoke(state())
    assert result['final_answer'] == 'Deadline is 15 January [1].'
    assert store.calls[0][2] == {'source': 'a.txt'}
    result = build_graph(store, Model(plan()), Model('Invented [9].', 'Still invented [9].'), Model()).invoke(state())
    assert 'valid source references' in result['final_answer']
    assert 'Still invented' not in result['final_answer']
    assert 'Deadline is 15 January.' in result['final_answer']


def test_missing_citation_is_repaired_and_checked():
    store = Store()
    answer = Model('Deadline is 15 January.', 'Deadline is 15 January [1].')
    result = build_graph(store, Model(plan()), answer, Model(checker_response())).invoke(state())
    assert len(answer.calls) == 2
    assert result['final_answer'] == 'Deadline is 15 January [1].'


def test_retrieval_order_and_deduplication():
    store = Store()
    assert [d.metadata['passage'] for d in Retrieval(store).whole(['a.txt'])] == [1, 2]
    assert len(deduplicate(store.docs + store.docs)) == 2


def test_checker_rejects_unsupported_claim():
    result = dict(supported=False, unsupported_claims=['claim'], citation_issues=[], conflicts_found=[], recommended_action='revise', confidence=0.8)
    checked = EvidenceChecker(Model(json.dumps(result))).check({'retrieved_chunks': Store().docs, 'draft_answer': 'A claim [1].'})
    assert 'could not verify' in checked['final_answer']
    assert 'A claim' not in checked['final_answer']


def test_checker_invalid_response_is_not_reported_as_missing_evidence():
    checked = EvidenceChecker(Model('not json')).check({'retrieved_chunks': Store().docs, 'draft_answer': 'A claim [1].'})
    assert checked['final_answer'].startswith(CHECK_FAILED)
    assert checked['evidence_check_result']['recommended_action'] == 'check_failed'


def test_no_passages_is_distinct():
    checked = EvidenceChecker(Model()).check({'retrieved_chunks': [], 'draft_answer': 'No evidence'})
    assert checked['final_answer'] == NO_PASSAGES


def test_revision_is_checked_before_returning():
    rejected = json.dumps(dict(supported=False, unsupported_claims=['wrong'], citation_issues=[], conflicts_found=[], recommended_action='revise', confidence=0.8))
    class Generator:
        def generate(self, *args):
            return 'Deadline is 15 January [1].'
    checked = EvidenceChecker(Model(rejected, checker_response()), Generator()).check(
        {'retrieved_chunks': Store().docs, 'draft_answer': 'Wrong [1].', 'user_message': 'What is the deadline?'})
    assert checked['final_answer'] == 'Deadline is 15 January [1].'


def test_history_old_and_new(tmp_path, monkeypatch):
    import chat_history
    monkeypatch.setattr(chat_history, 'STORAGE_DIR', tmp_path)
    monkeypatch.setattr(chat_history, 'HISTORY_FILE', tmp_path / 'chats.json')
    chats = [{'id': '1', 'messages': [{'role': 'assistant', 'content': 'old'}, {'role': 'assistant', 'content': 'new', 'skill': 'general_chat'}]}]
    save_chats(chats)
    assert load_chats() == chats
