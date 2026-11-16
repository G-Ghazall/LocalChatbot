"""Offline routing baseline; no answer quality or semantic correctness is implied."""
import json
from pathlib import Path
from agent import fallback

records = json.loads(Path('evaluation.json').read_text(encoding='utf-8'))
correct = clarify = 0
for case in records:
    result = fallback(case['user_query'], case['mode'], case['selected_documents'])
    correct += result.skill == case['expected_skill']
    clarify += (result.skill == 'clarify_question') == case['should_request_clarification']
    print(f"{case['id']}: {result.skill} (expected {case['expected_skill']})")
print(f'Routing accuracy: {correct}/{len(records)}; clarification accuracy: {clarify}/{len(records)}')
print('Fact coverage, unsupported claims, citations and insufficient-evidence accuracy: not measured by this routing-only baseline.')
