import json
import ollama

def generate_synthetic_sst2(model_name, sentiment, label_id, n=500):
    prompt = f"""Generate {n} diverse movie review sentences with a clear {sentiment} sentiment.
Return ONLY a JSON list of strings. No explanation.
Example format: ["review 1", "review 2", ...]"""

    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": prompt}
    ])
    sentences = json.loads(response["message"]["content"])
    return [{"sentence": s, "label": label_id} for s in sentences]

def generate_synthetic_snli(model_name, relation, label_id, n=300):
    prompt = f"""Generate {n} premise-hypothesis pairs where the hypothesis {relation} the premise.
Return ONLY a JSON list of objects with keys "premise" and "hypothesis". Don't add any introductory sentences.
Example: [{{"premise": "A man is walking.", "hypothesis": "Someone is moving."}}]"""

    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": prompt}
    ])
    pairs = json.loads(response["message"]["content"])
    return [{"premise": p["premise"], "hypothesis": p["hypothesis"], "label": label_id} for p in pairs]
