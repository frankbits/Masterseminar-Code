import json
import ollama

def generate_synthetic_data(model_name, prompt, n=500, max_retries=3):
    for attempt in range(max_retries):
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        print(f"LLM response (attempt {attempt + 1}): {response['message']['content']}")
        try:
            data = json.loads(response["message"]["content"])
            if isinstance(data, list) and len(data) >= n:
                return data[:n]
            else:
                print(f"Warning: LLM requested to generate {n} items but only returned {len(data)}. Retrying...")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}. Retrying...")

    raise ValueError(f"Failed to generate sufficient synthetic data after {max_retries} attempts.")

def generate_synthetic_sst2(model_name, sentiment, label_id, n=500, max_retries=3):
    prompt = f"""Generate {n} diverse movie review sentences with a clear {sentiment} sentiment.
Return ONLY a JSON list of strings. No explanation.
Example format: ["review 1", "review 2", ...]"""

    print(f"Generating {n} synthetic SST-2 sentences with {sentiment} sentiment using model '{model_name}'...")
    sentences = generate_synthetic_data(model_name, prompt, n, max_retries)

    return [{"sentence": s, "label": label_id} for s in sentences]

def generate_synthetic_snli(model_name, relation, label_id, n=300, max_retries=3):
    prompt = f"""Generate {n} premise-hypothesis pairs where the hypothesis {relation} the premise.
Return ONLY a JSON list of objects with keys "premise" and "hypothesis". Don't add any introductory sentences.
Example: [{{"premise": "A man is walking.", "hypothesis": "Someone is moving."}}]"""

    print(f"Generating {n} synthetic SNLI pairs with relation '{relation}' using model '{model_name}'...")
    pairs = generate_synthetic_data(model_name, prompt, n, max_retries)

    return [{"premise": p["premise"], "hypothesis": p["hypothesis"], "label": label_id} for p in pairs]
