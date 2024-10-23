import json

def read_jsonl(file_path):
    summaries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                summaries.append(json.loads(line))
    return summaries