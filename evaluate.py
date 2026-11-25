"""Offline routing check (no model calls). Measures routing only, not answer quality."""
import json
from pathlib import Path

from agent import route

records = json.loads(Path(__file__).with_name('evaluation.json').read_text(encoding='utf-8'))
correct = 0
for case in records:
    skill, _ = route(case['user_query'], case['mode'], case['selected_documents'])
    correct += skill == case['expected_skill']
    print(f"{'OK  ' if skill == case['expected_skill'] else 'MISS'} {case['id']}: {skill} (expected {case['expected_skill']})")
print(f'Routing accuracy: {correct}/{len(records)}')
