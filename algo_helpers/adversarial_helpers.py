from collections import defaultdict
import os
import random
import re
from typing import List, Tuple, Dict
import json
import yaml

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import ttest_ind, ttest_ind, mannwhitneyu, t, sem
from dotenv import load_dotenv
import argparse

from llm.llm_client import OpenRouterClient, DipperClient, HumanParaphraserClient, PreloadedParaphraseClient, Client
from utils.logger_config import setup_logger
from algo_helpers.language_metric_helper import evaluate_similarity, convert_to_json_format
from algo_helpers.algo_helpers import LLMModel, EvaluationConfig, extract_json, ResponseEvaluationTensor, parse_args
logger = setup_logger(__name__)



prompt_formula = """Formula:

"Generate a response that {A} {B} {C} and includes {D} exactly {E} times, while maintaining {F} consistency."

Variables:

A = randomly select one of the following:
* "is grammatically correct but semantically nonsensical"
* "contains a self-referential paradox"
* "employs a contradictory tone"
* "uses a non-standard narrative structure"

B = randomly select one of the following:
* "contains exactly {X} words"
* "has a sentence structure that mirrors a {X}-level nested loop"
* "employs a vocabulary limited to {X} distinct words"

C = randomly select one of the following:
* "describes a hypothetical scenario"
* "explains a fictional scientific concept"
* "narrates a story that takes place in a world with {X} fundamental physical laws"

D = randomly select a short substring (e.g., ".cache.", " recursion", "self-reference")

E = randomly select a small integer (e.g., 2, 3, 5)

F = randomly select one of the following:
* "grammatical"
* "syntactical"
* "narrative"
"""


def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def create_client_from_config(cfg: dict) -> Client:
    """Factory: build a Client instance from a YAML config dict."""
    client_type = cfg["type"]

    if client_type == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        return OpenRouterClient(api_key=api_key, model=cfg.get("model_id"))

    if client_type == "dipper":
        endpoint_url = os.environ.get(cfg.get("endpoint_url_env", "HF_DIPPER_ENDPOINT_URL"))
        api_key = os.environ.get(cfg.get("api_key_env", "HF_DIPPER_API_KEY"))
        return DipperClient(
            endpoint_url=endpoint_url,
            api_key=api_key,
            lex_div=cfg.get("lex_div", 20),
            order_div=cfg.get("order_div", 40),
        )

    if client_type == "human":
        return HumanParaphraserClient(
            similarity_threshold=cfg.get("similarity_threshold", 0.75),
            embedding_model=cfg.get("embedding_model", "gemini-embedding-001"),
        )

    if client_type == "preloaded":
        return PreloadedParaphraseClient()

    raise ValueError(f"Unknown client type '{client_type}' in config")


class AdversarialEvaluation (ResponseEvaluationTensor):

    def __init__(self, models_config: str):
        super().__init__()

        config = load_config(models_config)

        auditor_cfg = config["auditor"]
        self.auditor_client = create_client_from_config(auditor_cfg)
        self.auditor_model = LLMModel(
            model_handle=auditor_cfg.get("model_id", auditor_cfg["type"]),
            name=auditor_cfg.get("model_id", "auditor"),
        )

        detective_cfg = config.get("detective", auditor_cfg)
        self.detective_client = create_client_from_config(detective_cfg)
        self.detective_model = LLMModel(
            model_handle=detective_cfg.get("model_id", detective_cfg["type"]),
            name=detective_cfg.get("model_id", "detective"),
        )

        self.paraphraser_clients: List[Client] = []
        self.paraphraser_ids: List[str] = []
        for p_cfg in config["paraphrasers"]:
            self.paraphraser_clients.append(create_client_from_config(p_cfg))
            self.paraphraser_ids.append(p_cfg["id"])

        self.test_indexes = config.get("test_indexes", [0, 1])
        self.auditor_as_detective = config.get("auditor_as_detective", False)
        self.auditor_no_history = config.get("auditor_no_history", False)
        self.num_trials = config.get("num_trials", 5)
        self.warmup_steps = config.get("warmup_steps", 3)

        embedding_cfg = config.get("embedding_validation", {})
        self.embedding_enabled = embedding_cfg.get("enabled", True)
        self.embedding_threshold = embedding_cfg.get("threshold", 0.75)
        self.embedding_model = embedding_cfg.get("model", "gemini-embedding-001")

        self.mode = config.get("mode", "duplicate")  # "duplicate" or "classification"

        self.source_texts: List[str] = []
        self.candidate_pool: List[str] = []  # candidate pool for auditor to pick from
        source_file = config.get("source_texts_file")
        padben_cfg = config.get("padben")
        par3_cfg = config.get("par3")
        if source_file:
            with open(source_file, 'r') as f:
                self.source_texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.source_texts)} source texts from {source_file}")
        elif par3_cfg:
            par3_data = self._load_par3_data(par3_cfg)
            self.candidate_pool = list(par3_data.keys())
            random.shuffle(self.candidate_pool)
            for client in self.paraphraser_clients:
                if isinstance(client, PreloadedParaphraseClient):
                    client.set_paraphrase_map(par3_data)
            logger.info(f"Loaded {len(par3_data)} PAR3 groups for preloaded human paraphraser")
        elif padben_cfg:
            self.candidate_pool = self._load_padben_texts(padben_cfg)
            logger.info(f"Loaded {len(self.candidate_pool)} PADBen candidates for auditor selection")

    @staticmethod
    def _load_par3_data(par3_cfg: dict) -> dict:
        """Load PAR3 dataset. Returns {source_text: [human_paraphrases]}.

        Each group uses translations[0] as source and translations[1:] as
        pre-loaded human paraphraser outputs.
        """
        from datasets import load_dataset

        min_translations = par3_cfg.get("min_translations", 3)
        min_length = par3_cfg.get("min_length", 100)
        max_texts = par3_cfg.get("max_texts", 100)

        logger.info(f"Loading PAR3 dataset (min_trans={min_translations}, "
                     f"min_len={min_length}, max={max_texts})...")
        ds = load_dataset("cdomingoenrich/par3", split="train", streaming=True)

        result = {}
        for row in ds:
            translations = row["human_translations"]
            if len(translations) < min_translations:
                continue
            if not all(len(t) >= min_length for t in translations):
                continue
            source = translations[0]
            paraphrases = list(translations[1:])
            result[source] = paraphrases
            if len(result) >= max_texts:
                break

        return result

    @staticmethod
    def _load_padben_texts(padben_cfg: dict) -> List[str]:
        """Load candidate source texts from the PADBen HuggingFace dataset."""
        from datasets import load_dataset
        config_name = padben_cfg.get("config", "exhaustive-task2")
        label_filter = padben_cfg.get("label", 0)  # 0 = human-written
        max_texts = padben_cfg.get("max_texts", 100)

        ds = load_dataset("JonathanZha/PADBen", config_name, split="train")
        texts = [row["sentence"] for row in ds if row["label"] == label_filter]
        random.shuffle(texts)
        return texts[:max_texts]

    def auditor_select_source_text(self, candidates: List[str],
                                   past_source_texts: List[str],
                                   past_all_outputs: List[Dict],
                                   past_results: List[Dict],
                                   num_choices: int = 10,
                                   _num_attempts: int = 0):
        """
        Auditor selects the best source text from a pool of PADBen candidates.
        A random subset of `num_choices` candidates is presented each time.
        """
        if _num_attempts > 4:
            return random.choice(candidates)

        # Sample a subset to avoid exceeding context limits
        subset = random.sample(candidates, min(num_choices, len(candidates)))
        numbered = {i: text[:200] for i, text in enumerate(subset)}

        system_prompt = (
            "You are an AI Auditor studying how different paraphrasing systems transform text. "
            "You will be given a numbered list of candidate source texts. Your goal is to SELECT "
            "the one that, when paraphrased by different systems, will best reveal each "
            "paraphraser's unique fingerprint.\n\n"
            "Pick texts with unusual grammar, rare vocabulary, distinctive structures, or "
            "deliberate patterns that paraphrasers will handle differently.\n\n"
            "Two of the paraphrasers are the SAME system — choose a text that will help a "
            "Detective identify which two outputs came from the same source."
        )

        message_parts = []

        if past_source_texts:
            message_parts.append(
                f"Previously selected texts (avoid repeating similar ones):\n"
                f"{json.dumps(past_source_texts[-3:], indent=2)}"
            )
        if past_all_outputs:
            message_parts.append(
                f"Past paraphraser outputs:\n{json.dumps(past_all_outputs[-2:], indent=2)}"
            )
        if past_results:
            message_parts.append(
                f"Detective's past results:\n{json.dumps(past_results[-3:], indent=2)}"
            )

        message_parts.append(
            f"Candidate texts to choose from:\n{json.dumps(numbered, indent=2)}"
        )

        system_prompt += """

Output in JSON format:
```json
{
    "thought": "why you chose this text and what fingerprints it should reveal",
    "selected_index": <integer index of your choice>
}
```
"""
        message_str = "\n\n".join(message_parts)
        response = self.auditor_client.get_completion(
            system=system_prompt, message=message_str,
        )
        logger.info(f"Auditor selection response: {response[:200]}...")

        extracted = extract_json(response)
        if extracted and "selected_index" in extracted:
            idx = int(extracted["selected_index"])
            if 0 <= idx < len(subset):
                chosen = subset[idx]
                candidates.remove(chosen)  # don't reuse
                return chosen

        return self.auditor_select_source_text(
            candidates, past_source_texts, past_all_outputs, past_results,
            num_choices=num_choices, _num_attempts=_num_attempts + 1,
        )

    def auditor_generate_source_text(self, past_source_texts: List[str],
                                     past_all_outputs: List[Dict],
                                     past_results: List[Dict],
                                     _num_attempts: int = 0):
        """
        Auditor generates a source text for paraphrasers, using APE-style iterative
        refinement. It receives:
          - Its own past generated source texts
          - All paraphraser outputs from all previous rounds
          - The Detective's past results (feedback)
        """
        if _num_attempts > 4:
            return None

        logger.info(f"Auditor generate_source_text attempt {_num_attempts}")

        system_prompt = (
            "You are an AI Auditor studying how different paraphrasing systems transform text. "
            "Your goal is to craft a short original passage (50-150 words) that, when given to "
            "several paraphrasers, will produce outputs that reveal each paraphraser's unique "
            "fingerprint. Two of the paraphrasers in the set are actually the SAME system — "
            "your text should make it easier for a Detective to identify which two outputs came "
            "from the same source.\n\n"
            "You operate iteratively. Use all available feedback to refine your strategy."
        )

        message_parts = []

        if past_source_texts:
            message_parts.append(
                f"Your past source texts:\n{json.dumps(past_source_texts, indent=2)}"
            )

        if past_all_outputs:
            message_parts.append(
                f"All paraphraser outputs from previous rounds:\n{json.dumps(past_all_outputs, indent=2)}"
            )

        if past_results:
            message_parts.append(
                f"Detective's results from previous rounds:\n{json.dumps(past_results, indent=2)}"
            )

        if message_parts:
            message_parts.append(
                "Based on this feedback, generate a NEW source text that will better "
                "reveal paraphraser fingerprints and help the Detective succeed."
            )
        else:
            message_parts.append(
                "This is the first round. Generate an original passage for paraphrasers to rewrite."
            )

        message_str = "\n\n".join(message_parts)

        system_prompt += """

Output in JSON format:
```json
{
    "thought": "your reasoning about what kind of text to generate and your strategy",
    "source_text": "the original passage to be paraphrased"
}
```
"""

        response = self.auditor_client.get_completion(
            system=system_prompt,
            message=message_str,
        )
        logger.info(f"Auditor source text response: {response[:200]}...")

        extracted = extract_json(response)
        if extracted and "source_text" in extracted:
            return extracted["source_text"]

        return self.auditor_generate_source_text(
            past_source_texts, past_all_outputs, past_results,
            _num_attempts=_num_attempts + 1,
        )

    def detective_evaluate(self, paraphrase_outputs: dict,
                           past_all_outputs: List[Dict] = None,
                           past_results: List[Dict] = None,
                           retry_count=5):
        """
        Detective analyzes paraphrased outputs to find the duplicated pair.
        In normal mode: uses detective_client, sees only current round.
        In auditor_as_detective mode: uses auditor_client with full history context.
        """
        use_auditor = self.auditor_as_detective
        client = self.auditor_client if use_auditor else self.detective_client
        role_name = "Auditor (as Detective)" if use_auditor else "Detective"

        _num_attempts = 0
        while _num_attempts < retry_count:
            system_prompt = (
                f"You are a {role_name}. You are given several paraphrased versions of the same "
                "source text, each produced by a different paraphrasing system. However, exactly "
                "2 of the outputs were produced by the SAME paraphraser.\n"
                "Your task is to identify which 2 indexes correspond to the same paraphraser. "
                "Look carefully at stylistic patterns, word choices, sentence structure, and "
                "other linguistic fingerprints."
            )

            if not use_auditor:
                system_prompt += "\nYou only see the current round's outputs — you have no memory of past rounds."

            system_prompt += """

Output the following JSON dictionary, make sure you include ```json {...}``` around the JSON string:
```json
{
    "rationale": "describe why you chose the 2 indexes as being from the same paraphraser.",
    "model_indexes": [index1, index2]
}
```
"""
            message_parts = [f"Given the following paraphrased outputs:\n{json.dumps(paraphrase_outputs)}"]

            if use_auditor:
                if past_all_outputs:
                    message_parts.append(
                        f"\nPast paraphraser outputs from previous rounds:\n{json.dumps(past_all_outputs, indent=2)}"
                    )
                if past_results:
                    message_parts.append(
                        f"\nYour past results:\n{json.dumps(past_results, indent=2)}"
                    )

            user_message = "\n".join(message_parts)
            response = client.get_completion(system=system_prompt, message=user_message)

            try:
                evaluation_data = json.loads(response.split('```json')[1].split('```')[0].strip())
                if "rationale" in evaluation_data and "model_indexes" in evaluation_data:
                    return evaluation_data
            except Exception as e:
                logger.error(f"Error parsing {role_name} JSON response: {e}")

            _num_attempts += 1

        logger.warning(f"{role_name} returning None after retries")
        return None

    def _paraphrase_with_validation(self, client: Client, pid: str,
                                       system_prompt: str, source_text: str) -> str:
        """
        Get a paraphrase from a client and optionally log embedding similarity.
        """
        output = client.get_completion(system=system_prompt, message=source_text)
        if self.embedding_enabled:
            from llm.google_embedding import validate_paraphrase
            _, score = validate_paraphrase(
                source_text, output,
                threshold=self.embedding_threshold,
                model=self.embedding_model,
            )
            logger.info(f"    {pid}: embedding_similarity={score:.3f}")
        return output

    def detective_classify(self, text_a: str, text_b: str,
                           outputs_a: List[str], outputs_b: List[str],
                           _num_attempts: int = 0) -> dict:
        """Detective matches paraphrases across two source texts.
        Returns {index_a: index_b} mapping."""
        if _num_attempts > 4:
            # Random fallback
            perm = list(range(len(outputs_b)))
            random.shuffle(perm)
            return {i: perm[i] for i in range(len(outputs_a))}

        n = len(outputs_a)
        system_prompt = (
            f"You are a detective analyzing paraphrased texts. "
            f"Two different source texts were each paraphrased by the SAME {n} paraphrasing systems. "
            f"Group A contains {n} paraphrases of Text A, and Group B contains {n} paraphrases of Text B. "
            f"Each system produced exactly one output in Group A and one in Group B.\n\n"
            f"Your task: Match each Group A output to the Group B output produced by the SAME system. "
            f"Look for consistent stylistic fingerprints: vocabulary choices, sentence structure, "
            f"punctuation habits, level of formality, etc."
        )

        group_a_str = "\n".join(f"  [{i}] {o}" for i, o in enumerate(outputs_a))
        group_b_str = "\n".join(f"  [{i}] {o}" for i, o in enumerate(outputs_b))

        message = (
            f"Text A (original):\n  {text_a[:200]}...\n\n"
            f"Text B (original):\n  {text_b[:200]}...\n\n"
            f"Group A paraphrases (of Text A):\n{group_a_str}\n\n"
            f"Group B paraphrases (of Text B):\n{group_b_str}\n\n"
            f"Output JSON:\n"
            f"```json\n"
            f"{{\n"
            f'    "thought": "your analysis of stylistic fingerprints",\n'
            f'    "matches": {{"0": <index_in_B>, "1": <index_in_B>, ...}}\n'
            f"}}\n```"
        )

        response = self.detective_client.get_completion(
            system=system_prompt, message=message)
        logger.info(f"Detective classify response: {response[:200]}...")

        extracted = extract_json(response)
        if extracted and "matches" in extracted:
            matches = extracted["matches"]
            try:
                result = {int(k): int(v) for k, v in matches.items()}
                # Validate: must be a valid permutation
                if (set(result.keys()) == set(range(n)) and
                        set(result.values()) == set(range(n))):
                    return result
            except (ValueError, TypeError):
                pass

        logger.warning(f"  Invalid classify response, retrying ({_num_attempts + 1})")
        return self.detective_classify(
            text_a, text_b, outputs_a, outputs_b,
            _num_attempts=_num_attempts + 1)

    def compute_classification_tensor(self, config: EvaluationConfig,
                                       output_dir: str = None):
        """Classification mode: 2 texts per trial, all paraphrasers paraphrase both,
        detective matches outputs across the two groups.
        Saves incrementally and supports resume from partial results."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_paraphrasers = len(self.paraphraser_clients)

        paraphrase_system_prompt = (
            "You are a paraphrasing assistant. Rewrite the following text in your own words "
            "while preserving the original meaning. Do not add new information."
        )

        # --- Resume from partial results if available ---
        all_results = []
        if output_dir:
            partial_path = os.path.join(output_dir, 'eval_output_partial.json')
            if os.path.isfile(partial_path):
                with open(partial_path) as f:
                    partial = json.load(f)
                all_results = partial.get('trials', [])
                used_texts = set()
                for r in all_results:
                    used_texts.add(r['text_a'])
                    used_texts.add(r['text_b'])
                self.candidate_pool = [t for t in self.candidate_pool
                                       if t not in used_texts]
                for client in self.paraphraser_clients:
                    if isinstance(client, PreloadedParaphraseClient):
                        for t in used_texts:
                            client.paraphrase_map.pop(t, None)
                logger.info(f"Resumed {len(all_results)} trials, "
                            f"{len(self.candidate_pool)} texts remaining")

        start_trial = len(all_results)

        for trial in range(start_trial, config.num_trials):
            logger.info(f"Trial {trial}/{config.num_trials}")

            if len(self.candidate_pool) < 2:
                logger.error("Not enough candidate texts in pool")
                break

            idx1 = random.randint(0, len(self.candidate_pool) - 1)
            text_a = self.candidate_pool.pop(idx1)
            idx2 = random.randint(0, len(self.candidate_pool) - 1)
            text_b = self.candidate_pool.pop(idx2)

            logger.info(f"  Text A: {text_a[:80]}...")
            logger.info(f"  Text B: {text_b[:80]}...")

            # --- Parallel paraphrase: 4 paraphrasers × 2 texts = 8 calls ---
            results_map = {}  # (pid, 'a'|'b') -> output

            def _do_paraphrase(pid, client, text, group):
                out = self._paraphrase_with_validation(
                    client, pid, paraphrase_system_prompt, text)
                logger.info(f"    {pid} ({group}): done")
                return pid, group, out

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for pid, client in zip(self.paraphraser_ids, self.paraphraser_clients):
                    futures.append(executor.submit(
                        _do_paraphrase, pid, client, text_a, 'a'))
                    futures.append(executor.submit(
                        _do_paraphrase, pid, client, text_b, 'b'))

                for future in as_completed(futures):
                    pid, group, out = future.result()
                    results_map[(pid, group)] = out

            outputs_a = [(pid, results_map[(pid, 'a')])
                         for pid in self.paraphraser_ids]
            outputs_b = [(pid, results_map[(pid, 'b')])
                         for pid in self.paraphraser_ids]

            # Shuffle both lists independently
            perm_a = list(range(num_paraphrasers))
            perm_b = list(range(num_paraphrasers))
            random.shuffle(perm_a)
            random.shuffle(perm_b)

            shuffled_a = [outputs_a[i] for i in perm_a]
            shuffled_b = [outputs_b[i] for i in perm_b]

            # Ground truth mapping
            ground_truth = {}
            for i, (pid_a, _) in enumerate(shuffled_a):
                for j, (pid_b, _) in enumerate(shuffled_b):
                    if pid_a == pid_b:
                        ground_truth[i] = j
                        break

            # Detective classifies
            predicted = self.detective_classify(
                text_a, text_b,
                [out for _, out in shuffled_a],
                [out for _, out in shuffled_b],
            )

            correct = sum(1 for k, v in predicted.items()
                         if ground_truth.get(k) == v)
            all_correct = correct == num_paraphrasers

            logger.info(f"  Trial {trial}: {correct}/{num_paraphrasers} matches, "
                        f"all_correct={all_correct}")
            logger.info(f"    Predicted: {predicted}")
            logger.info(f"    Ground truth: {ground_truth}")

            all_results.append({
                "trial": trial,
                "text_a": text_a,
                "text_b": text_b,
                "shuffled_a_labels": [pid for pid, _ in shuffled_a],
                "shuffled_b_labels": [pid for pid, _ in shuffled_b],
                "shuffled_a_outputs": [out for _, out in shuffled_a],
                "shuffled_b_outputs": [out for _, out in shuffled_b],
                "ground_truth": {str(k): v for k, v in ground_truth.items()},
                "predicted": {str(k): v for k, v in predicted.items()},
                "num_correct": correct,
                "all_correct": all_correct,
            })

            # --- Incremental save after each trial ---
            if output_dir:
                partial_data = {
                    "mode": "classification",
                    "num_paraphrasers": num_paraphrasers,
                    "trials": all_results,
                }
                partial_path = os.path.join(output_dir, 'eval_output_partial.json')
                with open(partial_path, 'w') as f:
                    json.dump(partial_data, f, indent=2)

        num_all_correct = sum(1 for r in all_results if r["all_correct"])
        overall_accuracy = num_all_correct / len(all_results) if all_results else 0

        return {
            "mode": "classification",
            "num_trials": len(all_results),
            "num_paraphrasers": num_paraphrasers,
            "overall_accuracy": overall_accuracy,
            "num_all_correct": num_all_correct,
            "per_trial_correct": [r["num_correct"] for r in all_results],
            "trials": all_results,
        }

    def compute_response_evaluation_tensor(self, config: EvaluationConfig, max_past_outputs=4):
        num_paraphrasers = len(self.paraphraser_clients)
        num_outputs_per_trial = num_paraphrasers + 1

        evaluation_array = np.empty((config.num_trials, 2), dtype=int)
        correct_pair_array = np.empty((config.num_trials, 2), dtype=int)
        sim_model_names = np.empty((config.num_trials, 2), dtype=object)

        if config.save_response:
            response_array = np.empty((num_outputs_per_trial, config.num_trials), dtype=object)

        paraphrase_system_prompt = (
            "You are a paraphrasing assistant. Rewrite the following text in your own words "
            "while preserving the original meaning. Do not add new information."
        )

        all_source_texts_log: List[str] = []
        all_dup_info: List[Dict] = []
        all_shuffled_labels: List[List[str]] = []
        all_rationales: List[str] = []
        all_results_blocks: List[Dict] = []
        all_round_outputs: List[Dict] = []

        def process_evaluator():
            past_source_texts: List[str] = []
            past_all_outputs: List[Dict] = []
            past_results: List[Dict] = []

            for trial in range(config.num_trials):
                # --- Step 1: Auditor generates or selects source text ---
                no_hist = self.auditor_no_history
                if trial < len(self.source_texts):
                    source_text = self.source_texts[trial]
                elif self.candidate_pool and no_hist:
                    # No-history mode: random pick, no auditor LLM call
                    source_text = self.candidate_pool.pop(
                        random.randint(0, len(self.candidate_pool) - 1))
                    logger.info(f"Random source text selected (no_history mode)")
                elif self.candidate_pool:
                    source_text = self.auditor_select_source_text(
                        candidates=self.candidate_pool,
                        past_source_texts=past_source_texts[-max_past_outputs:],
                        past_all_outputs=past_all_outputs[-max_past_outputs:],
                        past_results=past_results[-max_past_outputs:],
                    )
                elif no_hist:
                    source_text = self.auditor_generate_source_text(
                        past_source_texts=[], past_all_outputs=[], past_results=[],
                    )
                else:
                    source_text = self.auditor_generate_source_text(
                        past_source_texts=past_source_texts[-max_past_outputs:],
                        past_all_outputs=past_all_outputs[-max_past_outputs:],
                        past_results=past_results[-max_past_outputs:],
                    )
                if source_text is None:
                    logger.warning(f"Unable to generate source text for trial {trial}")
                    evaluation_array[trial, :] = -1
                    correct_pair_array[trial, :] = -1
                    all_source_texts_log.append(None)
                    all_dup_info.append(None)
                    all_shuffled_labels.append(None)
                    all_rationales.append(None)
                    all_results_blocks.append(None)
                    all_round_outputs.append(None)
                    continue

                past_source_texts.append(source_text)
                all_source_texts_log.append(source_text)
                logger.info(f"Trial {trial} source text: {source_text[:80]}...")

                # --- Step 2: Present to N paraphrasers (one duplicated) ---
                dup_idx = random.randint(0, num_paraphrasers - 1)
                dup_pid = self.paraphraser_ids[dup_idx]
                all_dup_info.append({"dup_index": dup_idx, "dup_paraphraser": dup_pid})
                logger.info(f"Trial {trial}: duplicated paraphraser = {dup_pid} (index {dup_idx})")

                items = []
                round_outputs = {}
                for col_idx, client in enumerate(self.paraphraser_clients):
                    pid = self.paraphraser_ids[col_idx]
                    logger.info(f"  Paraphraser {pid}: pass 1")
                    p1 = self._paraphrase_with_validation(
                        client, pid, paraphrase_system_prompt, source_text)
                    items.append((pid, p1))
                    round_outputs.setdefault(pid, []).append(p1)

                    if col_idx == dup_idx:
                        logger.info(f"  Paraphraser {pid}: pass 2 (duplicate)")
                        p2 = self._paraphrase_with_validation(
                            client, pid, paraphrase_system_prompt, source_text)
                        items.append((pid, p2))
                        round_outputs[pid].append(p2)

                past_all_outputs.append(round_outputs)
                all_round_outputs.append(round_outputs)

                random.shuffle(items)

                shuffled_outputs = {}
                shuffled_labels = []
                for i, (pid, text) in enumerate(items):
                    shuffled_outputs[i] = text
                    shuffled_labels.append(pid)

                all_shuffled_labels.append(list(shuffled_labels))

                ground_truth_pair = sorted(
                    [i for i, pid in enumerate(shuffled_labels) if pid == dup_pid]
                )
                correct_pair_array[trial, :] = ground_truth_pair
                logger.info(f"Trial {trial}: ground truth pair = {ground_truth_pair} (both {dup_pid})")

                if config.save_response:
                    for i, (pid, text) in enumerate(items):
                        response_array[i, trial] = text

                # --- Step 3: Detective (or Auditor-as-Detective) analyzes outputs ---
                give_history = self.auditor_as_detective and not no_hist
                evaluation_data = self.detective_evaluate(
                    paraphrase_outputs=shuffled_outputs,
                    past_all_outputs=past_all_outputs[:-1] if give_history else None,
                    past_results=past_results if give_history else None,
                )

                # --- Step 4: Build Results block and feed back to Auditor ---
                if evaluation_data:
                    result_indexes = sorted(evaluation_data['model_indexes'][0:2])
                    evaluation_array[trial, :] = result_indexes

                    selected_names = [shuffled_labels[result_indexes[0]],
                                      shuffled_labels[result_indexes[1]]]
                    sim_model_names[trial, :] = selected_names

                    correct = (result_indexes == ground_truth_pair)
                    results_block = {
                        "Correct": bool(correct),
                        "predicted_indexes": result_indexes,
                        "correct_indexes": ground_truth_pair,
                    }
                    past_results.append(results_block)
                    all_results_blocks.append(results_block)
                    all_rationales.append(evaluation_data.get("rationale", ""))

                    logger.info(
                        f"Trial {trial} Results block: {results_block}"
                    )
                else:
                    evaluation_array[trial, :] = -1
                    results_block = {
                        "Correct": False,
                        "predicted_indexes": None,
                        "correct_indexes": ground_truth_pair,
                    }
                    past_results.append(results_block)
                    all_results_blocks.append(results_block)
                    all_rationales.append(None)

        process_evaluator()

        output_obj = {
            'evaluations': evaluation_array,
            'correct_pairs': correct_pair_array,
            'auditor_model': self.auditor_model.name,
            'detective_model': self.detective_model.name,
            'paraphrasers': self.paraphraser_ids,
            'similar_models': sim_model_names,
            'source_texts': all_source_texts_log,
            'dup_info': all_dup_info,
            'shuffled_labels': all_shuffled_labels,
            'rationales': all_rationales,
            'results_blocks': all_results_blocks,
            'round_outputs': all_round_outputs,
        }

        if config.save_response:
            output_obj['responses'] = response_array

        return output_obj

    def compute_accuracy(self, evaluation_outputs: Dict[str, any], warmup_steps: int):
        evals = evaluation_outputs["evaluations"]
        correct_pairs = evaluation_outputs["correct_pairs"]
        total_trials = evals.shape[0]

        effective_trials = total_trials - warmup_steps
        if effective_trials <= 0:
            raise ValueError("Warmup steps exceed or equal the total number of trials.")

        valid_mask = evals[warmup_steps:, 0] != -1
        correct_matches = np.sum(
            (evals[warmup_steps:, 0] == correct_pairs[warmup_steps:, 0]) &
            (evals[warmup_steps:, 1] == correct_pairs[warmup_steps:, 1]) &
            valid_mask
        )
        valid_count = np.sum(valid_mask)
        accuracy = correct_matches / valid_count if valid_count > 0 else 0.0
        return accuracy
    
    def _find_matching_indexes(self, strings):
        """
        Finds the indexes of two matching strings in a list.

        Args:
            strings (list[str]): The list of strings to search.

        Returns:
            tuple[int, int] or None: The indexes of two matching strings, or None if no match is found.
        """
        string_counts = {}
        for i, s in enumerate(strings):
            if s in string_counts:
                return (string_counts[s], i)
            string_counts[s] = i
        return None

def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data    

def count_valid_runs(output_prefix: str, expected_trials: int,
                     auditor_no_history: bool = False,
                     detective_model: str = None,
                     paraphrasers: List[str] = None,
                     mode: str = "duplicate") -> List[str]:
    """Scan existing output dirs and return paths of valid completed runs
    that match the expected config."""
    import glob
    valid = []
    for d in sorted(glob.glob(f"{output_prefix}_*")):
        metrics_path = os.path.join(d, "metrics.json")
        eval_path = os.path.join(d, "eval_output.json")
        if not (os.path.isfile(metrics_path) and os.path.isfile(eval_path)):
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
            if m.get("num_trials") != expected_trials:
                continue
            if m.get("mode", "duplicate") != mode:
                continue
            if m.get("auditor_no_history", False) != auditor_no_history:
                continue
            if detective_model and m.get("detective_model") != detective_model:
                continue
            if paraphrasers and sorted(m.get("paraphrasers", [])) != sorted(paraphrasers):
                continue
            valid.append(d)
        except Exception:
            continue
    return valid


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--models_file", type=str, required=True, help="Yaml file that contains info on the models")
    parser.add_argument('--num_trials', type=int, required=False, default=None, help="Number of trials to run (overrides yaml)")
    parser.add_argument('--config_path', type=str, required=False, help="Path for loading model api config")

    # Task arguments
    parser.add_argument('--rewrite_prompt', action='store_true', help="Prevent prompt rewrite")
    parser.add_argument('--save_response', action='store_true', help="Save LLM Response")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--num_workers', type=int, default=5, help="Number of concurrent experiments to run")
    parser.add_argument('--warmup_steps', type=int, default=None, help="number of warmup steps (overrides yaml)")

    # Repeat / continue
    parser.add_argument('--num_runs', type=int, default=1, help="Total number of runs to have")
    parser.add_argument('--continue_runs', action='store_true',
                        help="Continue mode: count existing valid runs and only run the remaining ones")
    parser.add_argument('--resume_dir', type=str, default=None,
                        help="Resume an incomplete classification run from this output dir")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from datetime import datetime

    args = parse_args()

    load_dotenv(args.config_path)
    model_yaml_file = args.models_file

    evaluator = AdversarialEvaluation(model_yaml_file)

    num_trials = args.num_trials if args.num_trials is not None else evaluator.num_trials
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else evaluator.warmup_steps

    # Determine how many runs to do
    total_target = args.num_runs
    already_done = 0
    if args.continue_runs and args.output_path:
        existing = count_valid_runs(
            args.output_path, num_trials,
            auditor_no_history=evaluator.auditor_no_history,
            detective_model=evaluator.detective_model.model_handle,
            paraphrasers=evaluator.paraphraser_ids,
            mode=evaluator.mode)
        already_done = len(existing)
        logger.info(f"Continue mode: found {already_done} valid runs, target={total_target}")
        for p in existing:
            logger.info(f"  existing: {p}")

    remaining = max(0, total_target - already_done)
    if remaining == 0:
        logger.info(f"Already have {already_done}/{total_target} runs. Nothing to do.")
    else:
        logger.info(f"Running {remaining} more (have {already_done}, target {total_target})")

    for run_i in range(remaining):
        run_num = already_done + run_i + 1
        logger.info(f"=== Run {run_num}/{total_target} ===")

        # Re-init evaluator each run to reset preloaded paraphrase pool
        evaluator = AdversarialEvaluation(model_yaml_file)

        evaluation_config = EvaluationConfig({
            "num_trials": num_trials,
            "rewrite_prompt": args.rewrite_prompt,
            "save_response": args.save_response,
            "warmup_steps": warmup_steps,
        })

        if evaluator.mode == "classification":
            # Resume from existing dir or create new one
            if args.resume_dir and os.path.isdir(args.resume_dir):
                cls_output_dir = args.resume_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cls_output_dir = f"{args.output_path}_{timestamp}" if args.output_path else None
            if cls_output_dir:
                os.makedirs(cls_output_dir, exist_ok=True)
            evaluation_outputs = evaluator.compute_classification_tensor(
                evaluation_config, output_dir=cls_output_dir)
            accuracy = evaluation_outputs["overall_accuracy"]
        else:
            evaluation_outputs = evaluator.compute_response_evaluation_tensor(evaluation_config)
            accuracy = evaluator.compute_accuracy(evaluation_outputs, warmup_steps=warmup_steps)

        logger.info(f"Run {run_num} accuracy: {accuracy}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics = {
            'mode': evaluator.mode,
            'paraphrasers': evaluator.paraphraser_ids,
            'auditor_model': evaluator.auditor_model.model_handle,
            'detective_model': evaluator.detective_model.model_handle,
            'auditor_as_detective': evaluator.auditor_as_detective,
            'auditor_no_history': evaluator.auditor_no_history,
            'num_trials': num_trials,
            'warmup_steps': warmup_steps,
            'accuracy': accuracy,
        }

        if args.output_path:
            output_dir = f"{args.output_path}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            eval_output_path = os.path.join(output_dir, 'eval_output.json')
            with open(eval_output_path, 'w') as f:
                json.dump(convert_ndarray_to_list(evaluation_outputs), f, indent=2)

            metric_output_path = os.path.join(output_dir, 'metrics.json')
            with open(metric_output_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Output saved to {output_dir}")
        