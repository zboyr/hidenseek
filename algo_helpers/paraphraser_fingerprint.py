from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils.logger_config import setup_logger


logger = setup_logger(__name__)


def _post_openrouter_chat(
    api_key: str,
    endpoint: str,
    model_id: str,
    prompt: str,
    temperature: float,
    seed: int,
    timeout_seconds: int,
    max_retries: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if response.status_code >= 500:
                raise requests.HTTPError(f"Server error {response.status_code}: {response.text}", response=response)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(min(8.0, float(2 ** attempt)))


DEFAULT_PARAPHRASE_PROMPT_TEMPLATE = """You are a paraphrasing system.

Rewrite the source text while preserving meaning and all key entities.
Constraints:
- Keep semantic meaning unchanged.
- Keep important entities, names, and numbers.
- Do not add new facts.
- Target length must stay within +/- {length_tolerance_percent}% of the source length.
- Style constraint: {style_constraint}
- Syntax constraint: {syntax_constraint}

Output only the rewritten text body.

Source text:
{source_text}
"""


@dataclass
class ParaphraseConstraints:
    length_tolerance_percent: int
    style_constraint: str
    syntax_constraint: str


class Paraphraser(ABC):
    def __init__(self, paraphraser_id: str):
        self.paraphraser_id = paraphraser_id
        self._last_metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self.supports_parallel_requests = False

    @abstractmethod
    def paraphrase(
        self,
        text: str,
        seed: int,
        temperature: float,
        constraints: ParaphraseConstraints,
    ) -> str:
        raise NotImplementedError

    def get_last_metadata(self) -> Dict[str, Any]:
        local_meta = getattr(self._thread_local, "last_metadata", None)
        if isinstance(local_meta, dict):
            return dict(local_meta)
        return dict(self._last_metadata)

    def _set_last_metadata(self, metadata: Dict[str, Any]) -> None:
        self._last_metadata = dict(metadata)
        self._thread_local.last_metadata = dict(metadata)


class OpenRouterParaphraser(Paraphraser):
    def __init__(
        self,
        paraphraser_id: str,
        model_id: str,
        api_key: Optional[str] = None,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ):
        super().__init__(paraphraser_id=paraphraser_id)
        self.supports_parallel_requests = True
        self.model_id = model_id
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterParaphraser")
        self.api_key = resolved_api_key

    def paraphrase(
        self,
        text: str,
        seed: int,
        temperature: float,
        constraints: ParaphraseConstraints,
    ) -> str:
        if self.api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterParaphraser")
        prompt = DEFAULT_PARAPHRASE_PROMPT_TEMPLATE.format(
            source_text=text,
            length_tolerance_percent=constraints.length_tolerance_percent,
            style_constraint=constraints.style_constraint,
            syntax_constraint=constraints.syntax_constraint,
        )
        body = _post_openrouter_chat(
            api_key=self.api_key,
            endpoint=self.endpoint,
            model_id=self.model_id,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        content = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        self._set_last_metadata(
            {
            "provider": "openrouter",
            "model_id": self.model_id,
            "seed": seed,
            "temperature": temperature,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            }
        )
        return content


class RemoteDipperParaphraser(Paraphraser):
    def __init__(
        self,
        paraphraser_id: str,
        model_id: str,
        lex_div: int,
        order_div: int,
        api_key: Optional[str] = None,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ):
        super().__init__(paraphraser_id=paraphraser_id)
        self.supports_parallel_requests = True
        self.model_id = model_id
        self.lex_div = lex_div
        self.order_div = order_div
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for RemoteDipperParaphraser")
        self.api_key = resolved_api_key

    def paraphrase(
        self,
        text: str,
        seed: int,
        temperature: float,
        constraints: ParaphraseConstraints,
    ) -> str:
        if self.api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for RemoteDipperParaphraser")
        prompt = f"""You are a DIPPER-style paraphrasing system.

Rewrite the source text while preserving meaning and all key entities.
Controls:
- lexical diversity level: {self.lex_div}
- order diversity level: {self.order_div}

Constraints:
- Keep semantic meaning unchanged.
- Keep important entities, names, and numbers.
- Do not add new facts.
- Target length must stay within +/- {constraints.length_tolerance_percent}% of the source length.
- Style constraint: {constraints.style_constraint}
- Syntax constraint: {constraints.syntax_constraint}

Output only the rewritten text body.

Source text:
{text}
"""
        body = _post_openrouter_chat(
            api_key=self.api_key,
            endpoint=self.endpoint,
            model_id=self.model_id,
            prompt=prompt,
            temperature=temperature,
            seed=seed,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        content = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        self._set_last_metadata(
            {
            "provider": "openrouter_dipper_remote",
            "model_id": self.model_id,
            "seed": seed,
            "temperature": temperature,
            "lex_div": self.lex_div,
            "order_div": self.order_div,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            }
        )
        return content


class RemoteTrueDipperParaphraser(Paraphraser):
    def __init__(
        self,
        paraphraser_id: str,
        endpoint_url: str,
        lex_div: int,
        order_div: int,
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ):
        super().__init__(paraphraser_id=paraphraser_id)
        self.supports_parallel_requests = True
        self.endpoint_url = endpoint_url
        self.lex_div = lex_div
        self.order_div = order_div
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        resolved_key = api_key or os.environ.get("HF_DIPPER_API_KEY") or os.environ.get("HF_TOKEN")
        if not resolved_key:
            raise ValueError("HF_DIPPER_API_KEY or HF_TOKEN is required for RemoteTrueDipperParaphraser")
        self.api_key = resolved_key

    @staticmethod
    def _extract_generated_text(body: Any) -> str:
        if isinstance(body, str):
            return body.strip()
        if isinstance(body, dict):
            if "generated_text" in body and body["generated_text"] is not None:
                return str(body["generated_text"]).strip()
            if "text" in body and body["text"] is not None:
                return str(body["text"]).strip()
            if "error" in body:
                raise RuntimeError(f"Remote DIPPER endpoint error: {body['error']}")
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"]).strip()
            return str(first).strip()
        raise RuntimeError(f"Unexpected remote DIPPER response format: {type(body)}")

    def paraphrase(
        self,
        text: str,
        seed: int,
        temperature: float,
        constraints: ParaphraseConstraints,
    ) -> str:
        lex_code = int(100 - self.lex_div)
        order_code = int(100 - self.order_div)
        dipper_input = f"lexical = {lex_code}, order = {order_code} <sent> {text.strip()} </sent>"

        payload = {
            "inputs": dipper_input,
            "parameters": {
                "do_sample": temperature > 0,
                "temperature": max(temperature, 1e-5),
                "max_new_tokens": 512,
                "seed": seed,
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        while True:
            attempt += 1
            try:
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 500:
                    raise requests.HTTPError(f"Server error {response.status_code}: {response.text}", response=response)
                response.raise_for_status()
                body = response.json()
                content = self._extract_generated_text(body)
                if not content:
                    raise RuntimeError("Remote DIPPER endpoint returned empty text")
                self._set_last_metadata(
                    {
                        "provider": "remote_true_dipper",
                        "endpoint_url": self.endpoint_url,
                        "seed": seed,
                        "temperature": temperature,
                        "lex_div": self.lex_div,
                        "order_div": self.order_div,
                        "length_tolerance_percent": constraints.length_tolerance_percent,
                    }
                )
                return content
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(8.0, float(2 ** attempt)))


class DipperParaphraser(Paraphraser):
    def __init__(
        self,
        paraphraser_id: str,
        model_name: str,
        lex_div: int,
        order_div: int,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(paraphraser_id=paraphraser_id)
        self.model_name = model_name
        self.lex_div = lex_div
        self.order_div = order_div
        self.tokenizer_name = tokenizer_name or "google/t5-v1_1-xxl"
        self.device = device
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer
        except Exception as exc:
            self._load_error = (
                "transformers is required for DipperParaphraser. "
                "Install transformers and a compatible torch runtime."
            )
            raise ImportError(self._load_error) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self._tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_name)

        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device:
            self._model = self._model.to(self.device)

    def paraphrase(
        self,
        text: str,
        seed: int,
        temperature: float,
        constraints: ParaphraseConstraints,
    ) -> str:
        self._lazy_load()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("DIPPER model/tokenizer not loaded")
        tokenizer: Any = self._tokenizer
        model: Any = self._model
        generation_input = (
            f"lexical = {self.lex_div}, order = {self.order_div} "
            f"<sent> {text.strip()} </sent>"
        )

        encoded = tokenizer(
            generation_input,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        generator_seeded = False
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            generator_seeded = True
        except Exception:
            generator_seeded = False

        with_input = {k: v.to(model.device) for k, v in encoded.items()}
        outputs = model.generate(
            **with_input,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            max_new_tokens=512,
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if decoded.startswith("paraphrase:"):
            decoded = decoded[len("paraphrase:") :].strip()

        self._set_last_metadata(
            {
                "provider": "local_dipper",
                "model_id": self.model_name,
                "tokenizer_name": self.tokenizer_name,
                "seed": seed,
                "temperature": temperature,
                "lex_div": self.lex_div,
                "order_div": self.order_div,
                "seed_applied": generator_seeded,
                "input_token_count": int(with_input["input_ids"].shape[-1]),
            }
        )
        return decoded


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle)


def load_source_texts(config: Dict[str, Any]) -> List[str]:
    inline_texts = config.get("source_texts", [])
    source_path = config.get("source_text_path")
    loaded: List[str] = []

    if source_path:
        with open(source_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    loaded.append(text)

    loaded.extend([text.strip() for text in inline_texts if isinstance(text, str) and text.strip()])
    if not loaded:
        raise ValueError("No source texts found. Provide source_text_path or source_texts in config.")
    return loaded


def build_paraphrasers(config: Dict[str, Any]) -> List[Paraphraser]:
    paraphrasers: List[Paraphraser] = []
    for spec in config.get("paraphrasers", []):
        paraphraser_type = spec.get("type")
        paraphraser_id = spec["id"]

        if paraphraser_type == "openrouter":
            paraphrasers.append(
                OpenRouterParaphraser(
                    paraphraser_id=paraphraser_id,
                    model_id=spec["model_id"],
                    api_key=spec.get("api_key"),
                    timeout_seconds=int(spec.get("timeout_seconds", 90)),
                    max_retries=int(spec.get("max_retries", 3)),
                )
            )
        elif paraphraser_type == "dipper_remote":
            paraphrasers.append(
                RemoteDipperParaphraser(
                    paraphraser_id=paraphraser_id,
                    model_id=spec["model_id"],
                    lex_div=int(spec["lex_div"]),
                    order_div=int(spec["order_div"]),
                    api_key=spec.get("api_key"),
                    timeout_seconds=int(spec.get("timeout_seconds", 90)),
                    max_retries=int(spec.get("max_retries", 3)),
                )
            )
        elif paraphraser_type == "dipper":
            paraphrasers.append(
                DipperParaphraser(
                    paraphraser_id=paraphraser_id,
                    model_name=spec.get("model_name", "kalpeshk2011/dipper-paraphraser-xxl"),
                    lex_div=int(spec["lex_div"]),
                    order_div=int(spec["order_div"]),
                    tokenizer_name=spec.get("tokenizer_name"),
                    device=spec.get("device"),
                )
            )
        elif paraphraser_type == "dipper_remote_true":
            endpoint_url = spec.get("endpoint_url")
            endpoint_url_env = spec.get("endpoint_url_env")
            if not endpoint_url and endpoint_url_env:
                endpoint_url = os.environ.get(str(endpoint_url_env))
            if not endpoint_url:
                raise ValueError(
                    f"Missing endpoint URL for {paraphraser_id}. Set endpoint_url or env var from endpoint_url_env."
                )

            api_key = spec.get("api_key")
            api_key_env = spec.get("api_key_env")
            if api_key is None and api_key_env:
                api_key = os.environ.get(str(api_key_env))

            paraphrasers.append(
                RemoteTrueDipperParaphraser(
                    paraphraser_id=paraphraser_id,
                    endpoint_url=str(endpoint_url),
                    lex_div=int(spec["lex_div"]),
                    order_div=int(spec["order_div"]),
                    api_key=api_key,
                    timeout_seconds=int(spec.get("timeout_seconds", 120)),
                    max_retries=int(spec.get("max_retries", 3)),
                )
            )
        else:
            raise ValueError(f"Unsupported paraphraser type: {paraphraser_type}")

    if not paraphrasers:
        raise ValueError("No paraphrasers configured.")
    return paraphrasers


class ParaphraserFingerprintEvaluation:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random = random.Random(int(config.get("random_seed", 0)))
        self.source_texts = load_source_texts(config)
        self.paraphrasers = build_paraphrasers(config)

        self.rounds = int(config.get("rounds", 5))
        self.samples_per_round = int(config.get("samples_per_round", 4))
        self.length_tolerance_percent = int(config.get("length_tolerance_percent", 15))
        self.temperature = float(config.get("temperature", 0.7))
        self.test_size = float(config.get("test_size", 0.2))
        self.max_workers = int(config.get("max_workers", 4))
        self.rate_limit_seconds = float(config.get("rate_limit_seconds", 0.0))
        self.output_path = config.get("output_path", "reports/paraphraser_fingerprint")
        self.output_with_timestamp = bool(config.get("output_with_timestamp", True))
        self.run_tag = str(config.get("run_tag", "run"))

        self.auditor_model = config.get("auditor_model", config.get("auditor_model_id", "openai/gpt-5.2"))
        if self.auditor_model and not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required when auditor_model is configured")
        self.detective_model = str(config.get("detective_model", "logreg_charwb"))
        self.source_difficulty: Dict[str, float] = {text: 1.0 for text in self.source_texts}
        self.records: List[Dict[str, Any]] = []

    def _build_detective(self) -> Tuple[TfidfVectorizer, LogisticRegression]:
        if self.detective_model == "logreg_charwb":
            return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)), LogisticRegression(max_iter=3000)
        if self.detective_model == "logreg_word":
            return TfidfVectorizer(analyzer="word", ngram_range=(1, 2)), LogisticRegression(max_iter=3000)
        raise ValueError(f"Unsupported detective_model: {self.detective_model}")

    def _sample_constraints(self) -> ParaphraseConstraints:
        style_options = self.config.get(
            "style_constraints",
            [
                "neutral tone",
                "formal register",
                "concise prose",
                "descriptive phrasing",
            ],
        )
        syntax_options = self.config.get(
            "syntax_constraints",
            [
                "vary sentence openings",
                "prefer active voice",
                "use one semicolon at most",
                "mix simple and compound sentences",
            ],
        )
        return ParaphraseConstraints(
            length_tolerance_percent=self.length_tolerance_percent,
            style_constraint=self.random.choice(style_options),
            syntax_constraint=self.random.choice(syntax_options),
        )

    def _choose_source_texts(self) -> List[str]:
        weights = [self.source_difficulty[text] for text in self.source_texts]
        chosen = self.random.choices(
            population=self.source_texts,
            weights=weights,
            k=self.samples_per_round,
        )
        return chosen

    def _auditor_evolve_source(self, source_text: str) -> str:
        if not self.auditor_model:
            return source_text

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when auditor_model is configured")

        system_prompt = (
            "You are an Auditor evolving source text to maximize stylistic separability "
            "between paraphrasers. Preserve original facts and entities."
        )
        user_prompt = (
            "Rewrite this source text into a harder paraphrasing challenge while preserving meaning. "
            "Return only the revised source text.\n\n"
            f"{source_text}"
        )
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = _post_openrouter_chat(
            api_key=api_key,
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            model_id=str(self.auditor_model),
            prompt=prompt,
            temperature=0.7,
            seed=self.random.randint(1, 2_000_000_000),
            timeout_seconds=90,
            max_retries=3,
        )
        evolved = response["choices"][0]["message"]["content"]
        evolved_text = str(evolved).strip()
        if not evolved_text:
            raise RuntimeError("Auditor returned empty evolved source_text")
        return evolved_text

    def _generate_one_record(
        self,
        round_idx: int,
        source_text: str,
        paraphraser: Paraphraser,
        constraints: ParaphraseConstraints,
    ) -> Dict[str, Any]:
        seed = self.random.randint(1, 2_000_000_000)
        if paraphraser.supports_parallel_requests:
            paraphrase = paraphraser.paraphrase(
                text=source_text,
                seed=seed,
                temperature=self.temperature,
                constraints=constraints,
            )
            metadata = paraphraser.get_last_metadata()
        else:
            with paraphraser._lock:
                paraphrase = paraphraser.paraphrase(
                    text=source_text,
                    seed=seed,
                    temperature=self.temperature,
                    constraints=constraints,
                )
                metadata = paraphraser.get_last_metadata()
        metadata.update(
            {
                "seed": seed,
                "temperature": self.temperature,
                "style_constraint": constraints.style_constraint,
                "syntax_constraint": constraints.syntax_constraint,
                "length_tolerance_percent": constraints.length_tolerance_percent,
            }
        )
        return {
            "round": round_idx,
            "source_text": source_text,
            "paraphraser_id": paraphraser.paraphraser_id,
            "paraphrase": paraphrase,
            "metadata": metadata,
        }

    def _generate_round(self, round_idx: int) -> List[Dict[str, Any]]:
        round_records: List[Dict[str, Any]] = []
        selected_sources = self._choose_source_texts()

        jobs: List[Tuple[str, Paraphraser, ParaphraseConstraints]] = []
        for source_text in selected_sources:
            evolved_source = self._auditor_evolve_source(source_text)
            for paraphraser in self.paraphrasers:
                jobs.append((evolved_source, paraphraser, self._sample_constraints()))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self._generate_one_record, round_idx, source_text, paraphraser, constraints): (
                    source_text,
                    paraphraser.paraphraser_id,
                )
                for source_text, paraphraser, constraints in jobs
            }

            for future in as_completed(future_to_job):
                source_text, paraphraser_id = future_to_job[future]
                try:
                    round_records.append(future.result())
                except Exception as exc:
                    logger.warning(
                        f"Generation failed for paraphraser={paraphraser_id}, source={source_text[:40]}..., error={exc}"
                    )
                if self.rate_limit_seconds > 0:
                    time.sleep(self.rate_limit_seconds)

        return round_records

    def _detective_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(records) < 2:
            return {"accuracy": 0.0, "macro_f1": 0.0, "num_samples": len(records)}

        texts = [record["paraphrase"] for record in records]
        labels = [record["paraphraser_id"] for record in records]

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return {"accuracy": 0.0, "macro_f1": 0.0, "num_samples": len(records)}

        enough_per_label = True
        for label in unique_labels:
            if labels.count(label) < 2:
                enough_per_label = False
                break

        vectorizer, model = self._build_detective()
        features = vectorizer.fit_transform(texts)

        num_samples = len(records)
        num_classes = len(unique_labels)
        min_test_count = num_classes
        max_test_count = num_samples - num_classes
        requested_test_count = int(math.ceil(num_samples * self.test_size))

        can_stratify_split = enough_per_label and max_test_count >= min_test_count

        if can_stratify_split:
            test_count = max(min_test_count, requested_test_count)
            test_count = min(test_count, max_test_count)
            split_test_size: float | int
            split_test_size = test_count

            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=split_test_size,
                random_state=42,
                stratify=labels,
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            eval_labels = y_test
        else:
            model.fit(features, labels)
            y_pred = model.predict(features)
            eval_labels = labels

        accuracy = accuracy_score(eval_labels, y_pred)
        macro_f1 = f1_score(eval_labels, y_pred, average="macro")

        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "num_samples": len(records),
        }

    def _update_source_difficulty(self, records: List[Dict[str, Any]]) -> None:
        if len(records) < 3:
            return

        texts = [record["paraphrase"] for record in records]
        labels = [record["paraphraser_id"] for record in records]
        sources = [record["source_text"] for record in records]

        vectorizer, model = self._build_detective()
        features = vectorizer.fit_transform(texts)
        model.fit(features, labels)
        predicted = model.predict(features)

        for source_text, gold, pred in zip(sources, labels, predicted):
            if gold != pred:
                self.source_difficulty[source_text] = self.source_difficulty.get(source_text, 1.0) + 0.25
            else:
                self.source_difficulty[source_text] = max(
                    1.0,
                    self.source_difficulty.get(source_text, 1.0) - 0.05,
                )

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.output_path, exist_ok=True)
        run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if self.output_with_timestamp:
            safe_run_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.run_tag).strip("_")
            if not safe_run_tag:
                safe_run_tag = "run"
            suffix = f"{safe_run_tag}_{run_timestamp}"
            records_path = os.path.join(self.output_path, f"paraphrase_records_{suffix}.jsonl")
            metrics_path = os.path.join(self.output_path, f"metrics_{suffix}.json")
        else:
            records_path = os.path.join(self.output_path, "paraphrase_records.jsonl")
            metrics_path = os.path.join(self.output_path, "metrics.json")

        with open(records_path, "w", encoding="utf-8"):
            pass

        round_metrics: List[Dict[str, Any]] = []

        for round_idx in range(self.rounds):
            round_records = self._generate_round(round_idx=round_idx)
            self.records.extend(round_records)

            with open(records_path, "a", encoding="utf-8") as handle:
                for record in round_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            detective_result = self._detective_metrics(self.records)
            detective_result["round"] = round_idx
            round_metrics.append(detective_result)
            self._update_source_difficulty(self.records)

            logger.info(
                "round=%s num_records=%s accuracy=%.4f macro_f1=%.4f",
                round_idx,
                detective_result["num_samples"],
                detective_result["accuracy"],
                detective_result["macro_f1"],
            )

        output = {
            "task": "paraphraser_fingerprint",
            "paraphrasers": [p.paraphraser_id for p in self.paraphrasers],
            "auditor_model": self.auditor_model,
            "detective_model": self.detective_model,
            "run_timestamp": run_timestamp,
            "round_metrics": round_metrics,
            "final_accuracy": round_metrics[-1]["accuracy"] if round_metrics else 0.0,
            "final_macro_f1": round_metrics[-1]["macro_f1"] if round_metrics else 0.0,
            "records_path": records_path,
            "num_records": len(self.records),
        }

        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

        output["metrics_path"] = metrics_path
        return output


def run_paraphraser_fingerprint_experiment(config_path: str, env_path: Optional[str] = None) -> Dict[str, Any]:
    load_dotenv(env_path)
    config = load_config(config_path)
    evaluation = ParaphraserFingerprintEvaluation(config)
    return evaluation.run()
