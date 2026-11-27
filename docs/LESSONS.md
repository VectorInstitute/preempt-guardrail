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
- `Sanitizer(ner_model, key, tweak)` has no default key. The name permutation and FF3 money cipher are seeded only by `key:tweak`, so a published default would let anyone reverse surrogates. Use `preempt.generate_key()` for a fresh pair. Library tests pass a fixed key. `feat/pii-bench` still calls `Sanitizer` with no key and needs the same change; this branch does not contain that code.
- `Sanitizer` accepts a `Recognizer` directly and wraps it in `LegacyNERShim`. `preempt/__init__.py` re-exports `Sanitizer`, `generate_key`, `LegacyNERShim`, and the contracts, but not the recognizers, which load transformers or the `gliner` extra.
- `demo.ipynb` reads `PREEMPT_MODEL_PATH`, `PREEMPT_KEY`, and `PREEMPT_TWEAK` (see `.env.example`) and loads `model.onnx`, not the int8 graph. Re-execute it with `uv run --env-file .env --with nbconvert --with ipywidgets jupyter nbconvert --execute --inplace src/preempt/demo.ipynb`. Without `ipywidgets` a tqdm warning writes a local path into the outputs. The kernel cannot start inside the agent sandbox.
- Compare configurations with a paired bootstrap on per-prompt means. Claim a difference only when the confidence interval excludes zero.
- The generator model and the judge model must be from different families.
- Offline CI path: `MockSanitizer` plus a stub LLM, 10 items, under 60 seconds. Seed the dataset, each sanitizer draw, and the bootstrap. Content-address the LLM cache and store model name and version on every record. Log full prompts; keep them off CI stdout.
- Bench tests that must exist before a run is trusted: slot spans, normalized leakage (`85k` / `85,000` / `eighty-five thousand`), name rehydration (possessives, case, partial names), `FINAL_ANSWER` parsing, each `answer_fn`, threshold distance, bootstrap CIs, judge JSON schema, and a golden file of the rendered judge prompt.

## Gotchas & Edge Cases

- `12,000 dollars` is the money format NER misses: 15/119 found (failure 0.87). `$12,000` is 107/107 and `$12k` is 134/134.
- Integer ages: 201/210 found (failure 0.04). Full names: 408/470 found (failure 0.13).
- After sanitization at ε = 1, an original value remains in 145/360 prompts (0.40). Money accounts for 117/360 prompts; names account for 64/470 name spans. Those denominators differ — prompts versus spans.
- Look-alikes (`metrics.lookalikes`, 20 non-record lines): 3 spans, all `NAME` — "Jordan" as a destination, "Jordan" as a person asked for notes, and "Bill". Left untagged: Apple, "the bill for dinner", Mark 4:12, version 52, Route 66, iPhone 15, "the age of the universe", China, Turkey, Georgia.
- The PII-masking train download (110,790 conversation rows, 54 label types, about 85 MB) lives in `training-data/`, which `.gitignore` excludes. Nothing in this branch reads it. A fresh clone does not have it, and it sits outside `src/preempt`, so it cannot land in the wheel.
- On `model.onnx` (run 407944) the demo sentence "Ben Parker and John Doe" yields `Ben Parker` and misses `John Doe`, so "John Doe" is left in place. The same graph does detect `Hamid Aziz` and `Jose Santiago` in the longer passage, and those round-trip. FPE money loses thousands separators (`$10,000` is sanitized with no comma and decrypts to `$10000`).
- Long text is scanned in overlapping windows (`DistilBertRecognizer.stride`, default 128 content tokens; `GlinerRecognizer.overlap`, default 32 words). `tokenizers` `return_overflowing_tokens` only returns one extra chunk, so the windows are built from the untruncated offsets. An entity longer than the overlap can still be split. If a long text has no character offsets, recognition raises instead of returning a silent prefix.
- One value FF3 or the parser cannot encrypt is replaced with `[REDACTED]` (`preempt.sanitizer.REDACTED`). The row is still listed in `invalid`, and the other values in that text are still replaced. FF3 money appends `999999` (`MONEY_FPE_SUFFIX`) rather than prefixing it: a prefix made the right half of a round amount (`$1,000,000`) all zeros, and pyfpe raises `ValueError: string 000000 is not within base/radix`. Decrypt strips that suffix.
- `decrypt` restores only surrogates saved by the last `encrypt` on that instance. A name the recognizer missed, or an amount the model reply added, is left as written. `key_only=True` is the explicit path that inverts listed names and FPE amounts with the key alone. In that mode an amount FF3 rejects is skipped, so one value cannot fail the batch. `use_cache` is still accepted and does not change this. Ages stay stateful either way.
- Leave ordinary look-alikes alone this round. A later hard-negative slice is place names that are also given names.
- Unsupported, so absent from the dataset: ages as words, birth dates, birth years, decades ("early 50s"), relative ages ("I'll be 60 next month"); money as words, slang ("about 85 grand"), ranges, or relations ("2x my spouse's salary").
- `runs/sweep-v2/` is the earlier 90-prompt sweep. It is a different run from the 360-prompt detection pass.

## Tooling & Environment Quirks

- Package manager is `uv`. The environment is `.venv` at the repo root. Layout is `src`, which differs from Poetry's flat layout.
- `typos` splits on hyphens, so `look-alikes` is reported as `alikes`. `Mata` in `demo.ipynb` is a generated surname. Both are allowlisted in `typos.toml` under `extend-ignore-words-re`. A new notebook run can introduce another surname that the checker rejects.
- `uv sync` installs the ONNX runtime. An exported ONNX file runs without PyTorch or Optimum.
- `pyfpe` 0.10.3 requires `pycryptodome==3.10.1`. pip-audit flags that release (PYSEC-2026-1811, fixed in 3.19.1). `tool.uv.override-dependencies` lifts it. The FF3 round-trip tests pass on pycryptodome 4.x.
- Extras: `uv sync --extra torch`, `--extra gliner`, `--extra export`.
- Dev and test dependencies: `uv sync --dev`. Drop a group with `uv sync --no-group docs`.
- Branch stacks, checked out on that branch: `uv sync --group train`, `--group ner-bench`, or `--group pii-bench`.
- `[tool.uv]` overrides and `[tool.uv.sources]` are not in published metadata. A `pip install preempt-guardrail` still gets `pycryptodome==3.10.1` from `pyfpe` and PyPI `optimum-onnx` for the `export` extra.
- Hatchling puts every non-ignored file under `src/preempt` in the wheel, so `names_dataset.json` ships and `*.ipynb` is excluded explicitly. The sdist has an `include` list so plans, benchmarks, and docs stay out. Check with `uv build` and list `dist/`.
- `uv lock` fails inside the agent sandbox because it clones `optimum-onnx` with git. Run it unsandboxed.
- The training set, `ai4privacy/pii-masking-200k`, has a custom license, not an open one. Non-profits may use it for free. It prohibits altering Ai4Privacy's work "with the intention to distribute, sell, rent, license, re-license, or sublicense your own derivative". The model stays private until ai4privacy (licensing@ai4privacy.com) or Vector legal confirms that publishing it is allowed. Do not label it Apache-2.0.
- The run 407944 ONNX files and model card are on Hugging Face as `vector-institute/preempt-distilbert-pii` (private). Load with `DistilBertRecognizer("vector-institute/preempt-distilbert-pii", backend="onnx", onnx_file_name="model.int8.onnx")`. A private repo needs `hf auth login`. The first run, which downloads the files, printed the right spans and then aborted at interpreter exit with `recursive_mutex lock failed` (macOS, `hf-xet` 1.6.0). Cached runs, online or with `HF_HUB_OFFLINE=1`, exit cleanly.
- Hub smoke test: "Hamid Aziz is 42 and lent Jose Santiago $10,000. Email hamid@example.com." gives `NAME` spans for each first and last name separately, `AGE` 42, and `MONEY` 10,000, but no `EMAIL` span, despite EMAIL F1 0.9997 on the held-out set.

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
