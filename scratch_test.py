import json
import re

def _dedup_text(text: str) -> str:
    if not text or len(text) < 100:
        return text
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        key = re.sub(r'\s+', ' ', s.strip().lower())
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(s.strip())
    return ' '.join(unique)

with open(r"c:\Users\arham\OneDrive\Attachments\Documents\GitHub\Agentic-Rag\notebooks_data\6d87da49-f46\forest_data\trees\3M_2018_10K_faf10c53fc6f__part01.json", "r", encoding="utf-8") as f:
    data = json.load(f)

node_0019 = next(n for n in data["nodes"] if n["node_id"] == "0019")
text = node_0019["text"]

print("Original length:", len(text))
deduped = _dedup_text(text)
print("Deduped length:", len(deduped))
print("Deduped preview:")
print(deduped[:1000])
print("...")
print(deduped[4000:4500])
