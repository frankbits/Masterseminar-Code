import ollama, json

def generate_synthetic_sst2(label_name, label_id, n=500):
    prompt = f"""Generate {n} diverse movie review sentences with a clear {label_name} sentiment.
Return ONLY a JSON list of strings. No explanation.
Example format: ["review 1", "review 2", ...]"""

    response = ollama.chat(model="llama3.1", messages=[
        {"role": "user", "content": prompt}
    ])
    sentences = json.loads(response["message"]["content"])
    return [{"sentence": s, "label": label_id} for s in sentences]

def generate_synthetic_snli(relation, label_id, n=300):
    prompt = f"""Generate {n} premise-hypothesis pairs where the hypothesis {relation} the premise.
Return ONLY a JSON list of objects with keys "premise" and "hypothesis". Don't add any introductory sentences.
Example: [{{"premise": "A man is walking.", "hypothesis": "Someone is moving."}}]"""

    response = ollama.chat(model="llama3.1", messages=[
        {"role": "user", "content": prompt}
    ])
    pairs = json.loads(response["message"]["content"])
    return [{"premise": p["premise"], "hypothesis": p["hypothesis"], "label": label_id} for p in pairs]

SAMPLE_SIZE = 10 #TODO: increase this for synthetic datasets for fine-tuning BERT

# Generate synthetic SST2 data
synthetic_neg = generate_synthetic_sst2("negative", 0, SAMPLE_SIZE)
synthetic_pos = generate_synthetic_sst2("positive", 1, SAMPLE_SIZE)

# Generate synthetic SNLI data
synthetic_entailment = generate_synthetic_snli("entails", 0, 10)
synthetic_contradiction = generate_synthetic_snli("contradicts", 1, SAMPLE_SIZE)
synthetic_neutral = generate_synthetic_snli("is neutral with respect to", 2, SAMPLE_SIZE)

PRINT_SAMPLE_SIZE = 10

# Print first 5 generated synthetic datapoints of SST2
synthetic_neg_trimmed = synthetic_neg[:PRINT_SAMPLE_SIZE]
synthetic_pos_trimmed = synthetic_pos[:PRINT_SAMPLE_SIZE]
synthetic_neg_str = json.dumps(synthetic_neg_trimmed, indent=4)
synthetic_pos_str = json.dumps(synthetic_pos_trimmed, indent=4)
print("Synthetic negative (SST2):", synthetic_neg_str, sep="\n")
print("Synthetic positive (SST2):", synthetic_pos_str, sep="\n")

# Print first 5 generated synthetic datapoints of SNLI
synthetic_entailment_trimmed    = synthetic_entailment[:PRINT_SAMPLE_SIZE]
synthetic_contradiction_trimmed = synthetic_contradiction[:PRINT_SAMPLE_SIZE]
synthetic_neutral_trimmed       = synthetic_neutral[:PRINT_SAMPLE_SIZE]
synthetic_entailment_str    = json.dumps(synthetic_entailment_trimmed, indent=4)
synthetic_contradiction_str = json.dumps(synthetic_contradiction_trimmed, indent=4)
synthetic_neutral_trimmed   = json.dumps(synthetic_neutral_trimmed, indent=4)
print("Synthetic entailment (SNLI):", synthetic_entailment_str, sep="\n")
print("Synthetic contradiction (SNLI):", synthetic_contradiction_str, sep="\n")
print("Synthetic neutral (SNLI):", synthetic_neutral_trimmed, sep="\n")