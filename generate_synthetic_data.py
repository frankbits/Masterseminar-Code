import json
import time

import ollama

def generate_synthetic_data(model_name, prompt, n=500, max_attempts=3) -> (dict, dict):
    generation_log = {
        "config": {
            "model_name": model_name,
            "prompt": prompt,
            "n": n,
            "max_attempts": max_attempts,
        },
        "generation_time_seconds": 0.0,
        "attempts": 0,
    }

    start_time = time.perf_counter()

    for attempt in range(max_attempts):
        generation_log["attempts"] += 1
        response = ollama.chat(model=model_name, messages=[
            {"role": "user", "content": prompt}
        ])
        print(f"LLM response (attempt {attempt + 1}): {response['message']['content']}")
        try:
            data = json.loads(response["message"]["content"])
            if isinstance(data, list) and len(data) >= n:
                generation_log["generation_time_seconds"] = time.perf_counter() - start_time
                return data[:n], generation_log
            else:
                print(f"Warning: LLM requested to generate {n} items but only returned {len(data)}. Retrying...")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}. Retrying...")

    raise ValueError(f"Failed to generate sufficient synthetic data after {max_attempts} attempts.")

def generate_synthetic_sst2(model_name, sentiment, label_id, n=500, max_attempts=3):
    prompt = f"""Generate {n} diverse movie review sentences with a clear {sentiment} sentiment.
Return ONLY a JSON list of strings. No explanation.
Example format: ["review 1", "review 2", ...]"""

    print(f"Generating {n} synthetic SST-2 sentences with {sentiment} sentiment using model '{model_name}'...")
    sentences, generation_log = generate_synthetic_data(model_name, prompt, n, max_attempts)

    return [{"sentence": s, "label": label_id} for s in sentences], generation_log

def generate_synthetic_snli(model_name, relation, label_id, n=300, max_attempts=3):
    prompt = f"""Generate {n} premise-hypothesis pairs where the hypothesis {relation} the premise.
Return ONLY a JSON list of objects with keys "premise" and "hypothesis". Don't add any introductory sentences.
Example: [{{"premise": "A man is walking.", "hypothesis": "Someone is moving."}}]"""

    print(f"Generating {n} synthetic SNLI pairs with relation '{relation}' using model '{model_name}'...")
    pairs, generation_log = generate_synthetic_data(model_name, prompt, n, max_attempts)

    return [{"premise": p["premise"], "hypothesis": p["hypothesis"], "label": label_id} for p in pairs], generation_log
