"""
Interactive human paraphrasing tool using PADBen dataset.

The LLM auditor selects the best text from a pool of 100 PADBen candidates,
then the human paraphrases it, and quality is validated with Google Embedding.

Usage:
    python human_paraphrase.py
    python human_paraphrase.py --pool_size 100 --threshold 0.8 --rounds 3
"""

import os
import random
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv


def load_padben_pool(config_name="exhaustive-task2", label=0, pool_size=100):
    """Load a pool of candidate texts from PADBen dataset."""
    from datasets import load_dataset

    print(f"Loading PADBen ({config_name}), pool_size={pool_size}...")
    ds = load_dataset("JonathanZha/PADBen", config_name, split="train")
    texts = [row["sentence"] for row in ds if row["label"] == label]
    random.shuffle(texts)
    return texts[:pool_size]


def auditor_select(client, pool, past_selected, num_choices=10):
    """LLM auditor picks the best candidate from a random subset of the pool."""
    from algo_helpers.algo_helpers import extract_json

    subset = random.sample(pool, min(num_choices, len(pool)))
    numbered = {i: text[:200] for i, text in enumerate(subset)}

    system_prompt = (
        "You are an AI Auditor. You will be given candidate source texts. "
        "Select the one that will best reveal paraphraser fingerprints when "
        "rewritten — look for unusual grammar, rare vocabulary, distinctive "
        "structures, or deliberate patterns that different paraphrasers will "
        "handle differently."
    )

    message_parts = []
    if past_selected:
        message_parts.append(
            f"Previously selected (avoid similar ones):\n"
            f"{json.dumps(past_selected[-3:], indent=2)}"
        )
    message_parts.append(
        f"Candidates:\n{json.dumps(numbered, indent=2)}"
    )

    system_prompt += """

Output JSON:
```json
{
    "thought": "why this text will reveal fingerprints",
    "selected_index": <integer>
}
```
"""
    response = client.get_completion(
        system=system_prompt,
        message="\n\n".join(message_parts),
    )

    extracted = extract_json(response)
    if extracted and "selected_index" in extracted:
        idx = int(extracted["selected_index"])
        if 0 <= idx < len(subset):
            chosen = subset[idx]
            pool.remove(chosen)
            return chosen

    # Fallback: random pick
    chosen = random.choice(subset)
    pool.remove(chosen)
    return chosen


def run_session(pool_size=100, similarity_threshold=0.75,
                embedding_model="gemini-embedding-001",
                padben_config="exhaustive-task2", padben_label=0,
                output_dir="human_paraphrase_results", rounds=1,
                auditor_model="anthropic/claude-sonnet-4.6"):
    """Run an interactive human paraphrasing session with LLM auditor selection."""
    from llm.llm_client import OpenRouterClient
    from llm.google_embedding import validate_paraphrase

    pool = load_padben_pool(
        config_name=padben_config,
        label=padben_label,
        pool_size=pool_size,
    )

    auditor = OpenRouterClient(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        model=auditor_model,
    )

    results = []
    past_selected = []

    for round_num in range(1, rounds + 1):
        if rounds > 1:
            print(f"\n--- Round {round_num}/{rounds} ---")

        # --- LLM auditor selects from pool ---
        print(f"\n  Auditor is selecting from {len(pool)} candidates...")
        original = auditor_select(auditor, pool, past_selected)
        past_selected.append(original)

        print(f"\n{'='*60}")
        print("  PARAPHRASE THE FOLLOWING TEXT")
        print(f"{'='*60}")
        print(f"\n{original}\n")

        # --- Human paraphrases with embedding validation ---
        while True:
            paraphrase = input("Your paraphrase:\n> ").strip()
            if not paraphrase:
                print("  Empty input. Please enter your paraphrase.")
                continue

            print("\n  Validating with Google Embedding...")
            is_valid, score = validate_paraphrase(
                original, paraphrase,
                threshold=similarity_threshold,
                model=embedding_model,
            )

            print(f"  Semantic similarity: {score:.4f}  (threshold: {similarity_threshold})")

            if is_valid:
                print("  Paraphrase accepted!")
                break
            else:
                print("  Similarity too low - paraphrase drifted too far from the original.")
                retry = input("  Try again? (y/n): ").strip().lower()
                if retry != "y":
                    print("  Keeping current paraphrase despite low similarity.")
                    break

        result = {
            "round": round_num,
            "original": original,
            "paraphrase": paraphrase,
            "similarity_score": score,
            "is_valid": is_valid,
            "similarity_threshold": similarity_threshold,
            "padben_config": padben_config,
        }
        results.append(result)

        # Replenish pool if running low
        if len(pool) < 10:
            pool = load_padben_pool(
                config_name=padben_config,
                label=padben_label,
                pool_size=pool_size,
            )

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"human_paraphrase_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human paraphrasing with LLM auditor selection from PADBen"
    )
    parser.add_argument("--pool_size", type=int, default=100,
                        help="Number of PADBen candidates to load")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Minimum cosine similarity threshold")
    parser.add_argument("--embedding_model", type=str,
                        default="gemini-embedding-001",
                        help="Google embedding model name")
    parser.add_argument("--padben_config", type=str, default="exhaustive-task2",
                        help="PADBen dataset configuration")
    parser.add_argument("--padben_label", type=int, default=0,
                        help="Label filter (0=human, 1=machine)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of paraphrasing rounds")
    parser.add_argument("--auditor_model", type=str,
                        default="anthropic/claude-sonnet-4.6",
                        help="LLM model for auditor selection")
    parser.add_argument("--output_dir", type=str,
                        default="human_paraphrase_results",
                        help="Output directory for results")
    parser.add_argument("--config_path", type=str, default=".env",
                        help="Path to .env file")

    args = parser.parse_args()
    load_dotenv(args.config_path)

    run_session(
        pool_size=args.pool_size,
        similarity_threshold=args.threshold,
        embedding_model=args.embedding_model,
        padben_config=args.padben_config,
        padben_label=args.padben_label,
        output_dir=args.output_dir,
        rounds=args.rounds,
        auditor_model=args.auditor_model,
    )
