# HideNSeek

HideNSeek is a model-fidelity verification project. The current default experiment is
paraphraser fingerprint attribution (Auditor evolves hard paraphrase cases, Detective
classifies which paraphraser produced each rewrite).

## Environment

Create `.env` and set keys used by your configured paraphrasers:

```
OPENROUTER_API_KEY=your_openrouter_key
HF_DIPPER_ENDPOINT_URL=your_remote_dipper_endpoint
HF_DIPPER_API_KEY=your_remote_dipper_token
```

`OPENROUTER_API_KEY` is required for `OpenRouterParaphraser` entries in the config.
`OPENROUTER_API_KEY` is also used by Auditor when `auditor_model` is enabled. Missing key or auditor failure hard-fails the run.
`HF_DIPPER_ENDPOINT_URL` + `HF_DIPPER_API_KEY` are required for `dipper_remote_true` entries.

## Default task: paraphraser fingerprint

Edit `paraphraser_config.yaml` to define:
- paraphraser list (OpenRouter model IDs + remote DIPPER-style variants)
- detective and auditor model variables (`detective_model`, `auditor_model`)
- generation params (temperature, rounds, samples, concurrency)
- source text corpus path
- timestamped outputs (`output_with_timestamp: true`, optional `run_tag`)

Run:

```
python -m algo_helpers.algo_helpers --task paraphraser_fingerprint --paraphraser_config paraphraser_config.yaml --config_path .env
```

Artifacts are written under `reports/paraphraser_fingerprint`:
- `paraphrase_records_<run_tag>_<timestamp>.jsonl` (fields: source_text, paraphraser_id, paraphrase, metadata)
- `metrics_<run_tag>_<timestamp>.json` (final accuracy, macro-F1, per-round curve)

## Legacy tasks

Legacy model-level tasks (`relevance`, `lang_trend`, `model_duplication`) are still
available and require `--models_file`.
