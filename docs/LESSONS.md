# Lessons

Constraints and history for later sessions. Sources: `README.md`, `pii-bench/reports/limitations.md`, and the pre-Milestone-8 questions in `pii_sanitizer_benchmark_plan.md`.

## Architecture & Design Decisions

- Preempt-Guardrail removes `AGE`, `MONEY`, and `NAME` from the user prompt before the LLM call. Usage example: `src/preempt/demo.ipynb`.
- Training, NER evaluation, and the end-to-end benchmark stay on separate branches off `main`, each with its own `uv` group:
  - `feat/training` (`train`) — fine-tune the DistilBERT NER for `NAME`, `AGE`, `MONEY`.
  - `feat/ner-bench` (`ner-bench`) — detection accuracy, quantization, and latency.
  - `feat/pii-bench` (`pii-bench`) — utility and privacy across ε, verifiable tasks, and an LLM judge.
- The default install runs an exported ONNX graph. `DistilBertRecognizer` defaults to PyTorch and needs the `torch` extra. Pass `backend="onnx"` on a default install. GLiNER and checkpoint export are the `gliner` and `export` extras.
- Benchmark data is synthetic. Keep real personal data, including public chat corpora, out of ground-truth metrics. A later sanity-check corpus needs a license check and a separate PII relabel.
- Scored detection set: 36 templates × 10 fills = 360 prompts in `data/dataset.jsonl`. 240 are incidental (email drafts, general explanations). 120 are verifiable (retirement, age gates, savings rate, raise). Recognizer and sanitizer ran on all 360 at ε = 1 via `metrics.detection`.
- Values this round generates: integer ages; money as `$12,000`, `$12k`, or `12,000 dollars`.
- Partial recall uses IoU ≥ 0.5. Per-item rows are in `runs/detection_validation.jsonl`. Strict money F1 is 0.40 because the recognizer labels the digits inside `$12,000` while the gold span includes `$`. Partial money F1 is 0.83. Quote the partial figure for money detection.
- Compare configurations with a paired bootstrap on per-prompt means. Claim a difference only when the confidence interval excludes zero.
- The generator model and the judge model must be from different families.
- Offline CI path: `MockSanitizer` plus a stub LLM, 10 items, under 60 seconds. Seed the dataset, each sanitizer draw, and the bootstrap. Content-address the LLM cache and store model name and version on every record. Log full prompts; keep them off CI stdout.
- Bench tests that must exist before a run is trusted: slot spans, normalized leakage (`85k` / `85,000` / `eighty-five thousand`), name rehydration (possessives, case, partial names), `FINAL_ANSWER` parsing, each `answer_fn`, threshold distance, bootstrap CIs, judge JSON schema, and a golden file of the rendered judge prompt.

## Gotchas & Edge Cases

- `12,000 dollars` is the money format NER misses: 15/119 found (failure 0.87). `$12,000` is 107/107 and `$12k` is 134/134.
- Integer ages: 201/210 found (failure 0.04). Full names: 408/470 found (failure 0.13).
- After sanitization at ε = 1, an original value remains in 145/360 prompts (0.40). Money accounts for 117/360 prompts; names account for 64/470 name spans. Those denominators differ — prompts versus spans.
- Look-alikes (`metrics.lookalikes`, 20 non-record lines): 3 spans, all `NAME` — "Jordan" as a destination, "Jordan" as a person asked for notes, and "Bill". Left untagged: Apple, "the bill for dinner", Mark 4:12, version 52, Route 66, iPhone 15, "the age of the universe", China, Turkey, Georgia.
- Leave ordinary look-alikes alone this round. A later hard-negative slice is place names that are also given names.
- Unsupported, so absent from the dataset: ages as words, birth dates, birth years, decades ("early 50s"), relative ages ("I'll be 60 next month"); money as words, slang ("about 85 grand"), ranges, or relations ("2x my spouse's salary").
- `runs/sweep-v2/` is the earlier 90-prompt sweep. It is a different run from the 360-prompt detection pass.

## Tooling & Environment Quirks

- Package manager is `uv`. The environment is `.venv` at the repo root. Layout is `src`, which differs from Poetry's flat layout.
- `uv sync` installs the ONNX runtime. An exported ONNX file runs without PyTorch or Optimum.
- Extras: `uv sync --extra torch`, `--extra gliner`, `--extra export`.
- Dev and test dependencies: `uv sync --dev`. Drop a group with `uv sync --no-group docs`.
- Branch stacks, checked out on that branch: `uv sync --group train`, `--group ner-bench`, or `--group pii-bench`.

## Active & Pending Technical Debt

- Resolve before Milestone 8, when the real sanitizer is wired:
  1. Sanitizer transport (Python function, REST, or CLI), and whether it returns the substitution map and spans.
  2. ε range and parameterization, and the distance metric mLDP uses.
  3. Which generator and judge models to use (different families).
  4. Whether FPE name mapping preserves gender, culture, and length. That answer decides whether to add pronoun-consistency checks.
  5. Whether this round includes non-English prompts.
- Still unrun on the 360-prompt set: generator, judge, a full ε sweep, privacy attacks, human labels for the judge, bootstrap intervals, and per-type ablations.
- Open quality gaps from the ε = 1 detection pass: `N dollars` recognition (failure 0.87) and residual original values in 0.40 of prompts. Strict versus partial money spans still disagree on the `$`.
- Large runs need a dry-run cost estimate and a confirmation prompt. Pilot is 50 items, 2 ε values, k = 2, and that report is reviewed before the full run.
- Milestones 1–3 come first. Stop for review before spending LLM budget on the full run.
