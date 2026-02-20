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

from llm.llm_client import OpenRouterClient, DipperClient, Client
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
        self.num_trials = config.get("num_trials", 5)
        self.warmup_steps = config.get("warmup_steps", 3)

        self.source_texts: List[str] = []
        source_file = config.get("source_texts_file")
        if source_file:
            with open(source_file, 'r') as f:
                self.source_texts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.source_texts)} source texts from {source_file}")

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
                # --- Step 1: Auditor generates source text ---
                if trial < len(self.source_texts):
                    source_text = self.source_texts[trial]
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
                    p1 = client.get_completion(system=paraphrase_system_prompt, message=source_text)
                    items.append((pid, p1))
                    round_outputs.setdefault(pid, []).append(p1)

                    if col_idx == dup_idx:
                        logger.info(f"  Paraphraser {pid}: pass 2 (duplicate)")
                        p2 = client.get_completion(system=paraphrase_system_prompt, message=source_text)
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
                evaluation_data = self.detective_evaluate(
                    paraphrase_outputs=shuffled_outputs,
                    past_all_outputs=past_all_outputs[:-1] if self.auditor_as_detective else None,
                    past_results=past_results if self.auditor_as_detective else None,
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

    evaluation_config = EvaluationConfig({
        "num_trials": num_trials,
        "rewrite_prompt": args.rewrite_prompt,
        "save_response": args.save_response,
        "warmup_steps": warmup_steps,
    })

    evaluation_outputs = evaluator.compute_response_evaluation_tensor(evaluation_config)
    accuracy = evaluator.compute_accuracy(evaluation_outputs, warmup_steps=warmup_steps)

    logger.info(f"evaluation_outputs:")
    logger.info(evaluation_outputs)
    logger.info("accuracy:")
    logger.info(accuracy)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = {
        'paraphrasers': evaluator.paraphraser_ids,
        'auditor_model': evaluator.auditor_model.model_handle,
        'detective_model': evaluator.detective_model.model_handle,
        'auditor_as_detective': evaluator.auditor_as_detective,
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
        