import json
import time

import ollama


def ask_user_how_to_proceed(error, batch_num, attempt):
    """
    Called when a single batch has exhausted its automatic retries.
    Prints the error and asks the user in the console how to proceed.

    Returns one of: "retry", "skip", "abort"
    """
    print(f"\n[!] Batch {batch_num}, attempt {attempt}: generation/parsing failed.")
    print(f"    Error: {error}")
    while True:
        choice = input("    Proceed how? [r]etry / [s]kip this batch / [a]bort entire run: ").strip().lower()
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("a", "abort"):
            return "abort"
        print("    Please enter 'r', 's', or 'a'.")


def generate_synthetic_data(model_name, seed, prompt_fn, response_key, n=500, batch_size=25, max_attempts=3) -> (list, dict):
    """
    Generates `n` items in batches of `batch_size`, calling Ollama with
    format="json" for each batch.

    prompt_fn:     function(batch_n) -> prompt string for a batch of that size
    response_key:  expected JSON key for the list of items, e.g. "reviews" or "pairs"
                   (a bare top-level JSON list is also accepted as a fallback)
    max_attempts:  automatic retries per batch before asking the user in console

    Returns: (items, generation_log)
    """
    generation_log = {
        "config": {
            "model_name": model_name,
            "seed": seed,
            "prompt": prompt_fn(batch_size),
            "n": n,
            "batch_size": batch_size,
            "max_attempts": max_attempts,
        },
        "batches": [],          # per-batch timing / attempts / retries
        "total_retries": 0,
        "aborted": False,
        "generation_time_seconds": 0.0,
        "items_generated": 0,
    }

    all_items = []
    batch_num = 0
    start_time = time.perf_counter()

    while len(all_items) < n and not generation_log["aborted"]:
        remaining = n - len(all_items)
        batch_n = min(batch_size, remaining)
        prompt = prompt_fn(batch_n)

        batch_log = {"batch_num": batch_num, "requested": batch_n,
                     "attempts": 0, "retries": 0, "time_seconds": None, "received": 0}
        start_batch = time.perf_counter()

        attempt = 0
        items = []
        while True:
            attempt += 1
            batch_log["attempts"] = attempt
            # Derive a distinct, reproducible seed per individual API call.
            call_seed = seed + batch_num * (max_attempts + 1) + attempt

            response = ollama.chat(
                model=model_name,
                options={"seed": call_seed},
                format="json",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"]
            print(f"LLM response (batch {batch_num}, attempt {attempt}): {content[:200]}...")

            try:
                data = json.loads(content)
                if isinstance(data, dict) and response_key in data:
                    items = data[response_key]
                elif isinstance(data, list):
                    items = data
                else:
                    raise ValueError(
                        f"Expected key '{response_key}' or a bare list, got: "
                        f"{list(data.keys()) if isinstance(data, dict) else type(data)}"
                    )

                if len(items) < batch_n:
                    print(f"Note: batch {batch_num} requested {batch_n} items, "
                          f"received {len(items)}.")
                break  # success (even if slightly short - we just take what we got)

            except (json.JSONDecodeError, ValueError) as e:
                if attempt <= max_attempts:
                    generation_log["total_retries"] += 1
                    batch_log["retries"] += 1
                    print(f"Generation/parsing error: {e}. Retrying ({attempt}/{max_attempts})...")
                    continue
                else:
                    decision = ask_user_how_to_proceed(e, batch_num, attempt)
                    if decision == "retry":
                        generation_log["total_retries"] += 1
                        batch_log["retries"] += 1
                        attempt = 0  # grant a fresh round of automatic retries
                        continue
                    elif decision == "skip":
                        items = []
                        break
                    else:  # abort
                        generation_log["aborted"] = True
                        items = []
                        break

        batch_log["time_seconds"] = time.perf_counter() - start_batch
        batch_log["received"] = len(items)
        generation_log["batches"].append(batch_log)

        if generation_log["aborted"]:
            break

        all_items.extend(items)
        batch_num += 1

    generation_log["generation_time_seconds"] = time.perf_counter() - start_time
    generation_log["items_generated"] = len(all_items)

    return all_items[:n], generation_log


def generate_synthetic_sst2(model_name, seed, sentiment, label_id, n=500, batch_size=25, max_attempts=3):
    def prompt_fn(batch_n):
        return f"""Generate {batch_n} diverse movie review sentences with a clear {sentiment} sentiment.

Output requirements:
- Respond with ONLY a JSON object: {{"reviews": ["review 1", "review 2", ...]}}
- Do not include any preamble, explanation, summary, numbering, or commentary of any kind.
- Your entire response must be valid JSON, starting with {{ and ending with }}."""

    print(f"Generating {n} synthetic SST-2 sentences with {sentiment} sentiment "
          f"using model '{model_name}' (batch size {batch_size})...")
    sentences, generation_log = generate_synthetic_data(
        model_name, seed, prompt_fn, response_key="reviews",
        n=n, batch_size=batch_size, max_attempts=max_attempts,
    )

    return [{"sentence": s, "label": label_id} for s in sentences], generation_log


def generate_synthetic_snli(model_name, seed, relation, label_id, n=300, batch_size=25, max_attempts=3):
    def prompt_fn(batch_n):
        return f"""Generate {batch_n} premise-hypothesis pairs where the hypothesis {relation} the premise.

Output requirements:
- Respond with ONLY a JSON object: {{"pairs": [{{"premise": "...", "hypothesis": "..."}}, ...]}}
- Do not include any preamble, explanation, summary, numbering, or commentary of any kind.
- Your entire response must be valid JSON, starting with {{ and ending with }}.

Example item: {{"premise": "A man is walking.", "hypothesis": "Someone is moving."}}"""

    print(f"Generating {n} synthetic SNLI pairs with relation '{relation}' "
          f"using model '{model_name}' (batch size {batch_size})...")
    pairs, generation_log = generate_synthetic_data(
        model_name, seed, prompt_fn, response_key="pairs",
        n=n, batch_size=batch_size, max_attempts=max_attempts,
    )

    return [{"premise": p["premise"], "hypothesis": p["hypothesis"], "label": label_id} for p in pairs], generation_log