# Preempt-Guardrail

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/preempt-guardrail/actions/workflows/docs.yml)
[![codecov](https://codecov.io/github/VectorInstitute/preempt-guardrail/graph/badge.svg)](https://codecov.io/github/VectorInstitute/preempt-guardrail)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/preempt-guardrail)

Preempt-Guardrail removes age, money, and name values from a user prompt before the prompt is sent to an LLM API, and restores them in the reply.

## Installation

```bash
pip install preempt-guardrail
```

The package is imported as `preempt`. The NER model is not in the package. It is hosted on Hugging Face as [`vector-institute/preempt-distilbert-pii`](https://huggingface.co/vector-institute/preempt-distilbert-pii) and downloaded on first use. `PREEMPT_MODEL_PATH` can also point at a local copy (see [Configuration](#configuration)).

```python
from preempt import Sanitizer, generate_key
from preempt.distilbert_recognizer import DistilBertRecognizer

recognizer = DistilBertRecognizer(
    "vector-institute/preempt-distilbert-pii",
    backend="onnx",
    onnx_file_name="model.int8.onnx",
)
key, tweak = generate_key()
sanitizer = Sanitizer(recognizer, key, tweak)

sanitized, failed = sanitizer.encrypt(["Ben lent Adam $10,000."], entity="Name")
restored = sanitizer.decrypt(sanitized, entity="Name")
```

For a full example, see [`demo.ipynb`](https://github.com/VectorInstitute/preempt-guardrail/blob/main/src/preempt/demo.ipynb).

## Approach

Preempt-Guardrail follows [Prεεmpt](https://arxiv.org/abs/2504.05147) (Roy Chowdhury et al., NDSS 2026; [reference code](https://github.com/danshumaan/preempt)). The paper leaves a usable stand-in in the prompt instead of a placeholder. Format-preserving encryption covers values whose shape matters, such as SSNs and card numbers. Metric differential privacy covers values such as age, salary, and a specific name, and bounds how much one response can reveal about the true value. This package sanitizes age, money, and name. Names and amounts are format-preserving and reversible with the key and tweak. Ages are noised. The recognizer also marks email, phone, address, and id, and those spans are left unchanged.

The reference implementation finds those spans with Universal NER (7B) or Llama-3-8B-Instruct loaded in process. That model is too large to run as a pre-call step on a CPU service. This repository uses a fine-tuned DistilBERT token classifier instead. It is small enough to serve in process, exports to ONNX, and the INT8 graph adds only a few milliseconds per short prompt on a laptop CPU. Held-out quality stays within a fraction of a point of the full-precision graph.

### Training set

The classifier was fine-tuned on the PII-masking 200k train set, `training-data/pii_masking_200k_en_fr_de_train.json`. That file is gitignored and is not part of the package.

| | |
| --- | --- |
| Examples | 110,790 labeled conversations |
| Languages | English, French, and German |
| Label types | 54, including names, ages, money, emails, phones, street addresses, dates, and account and card numbers |

The served checkpoint scores seven span types: `NAME`, `AGE`, `MONEY`, `EMAIL`, `PHONE`, `ADDRESS`, and `ID`.

### Held-out benchmark (run 407944)

| Graph | Precision | Recall | F1 | False-negative rate |
| --- | --- | --- | --- | --- |
| FP32 ONNX | 0.9946 | 0.9923 | 0.9935 | 0.0077 |
| INT8 ONNX | 0.9949 | 0.9916 | 0.9932 | 0.0084 |

| Graph | Mean | p50 | p95 | p99 |
| --- | --- | --- | --- | --- |
| FP32 ONNX | 19.0 ms | 17.6 ms | 32.7 ms | 44.6 ms |
| INT8 ONNX | 7.1 ms | 6.6 ms | 11.9 ms | 16.0 ms |

Per-label scores on the INT8 graph (29,651 spans in total). Money is the weakest label. Its recall is 0.954, and INT8 does not change that recall.

| Label | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| NAME | 10,203 | 0.9935 | 0.9939 | 0.9937 |
| AGE | 1,331 | 0.9902 | 0.9857 | 0.9880 |
| MONEY | 1,399 | 0.9852 | 0.9535 | 0.9691 |
| EMAIL | 1,788 | 1.0000 | 0.9994 | 0.9997 |
| PHONE | 1,231 | 1.0000 | 0.9968 | 0.9984 |
| ADDRESS | 7,610 | 0.9949 | 0.9908 | 0.9928 |
| ID | 6,089 | 0.9980 | 0.9952 | 0.9966 |

## Configuration

The demo reads the model path and the cipher key from the environment. Copy the example and fill it in:

```bash
cp .env.example .env
```

| Variable | Meaning |
| --- | --- |
| `PREEMPT_MODEL_PATH` | Hugging Face model id, or a local directory that contains `model.onnx` |
| `PREEMPT_KEY` | Hex FF3 key: 32, 48, or 64 hex characters (128, 192, or 256 bits) |
| `PREEMPT_TWEAK` | Hex FF3 tweak: 16 hex characters (64 bits) |

`.env` is gitignored. Anyone with the key and tweak can reverse FPE names and amounts. Generate a pair with:

```bash
uv run python -c "from preempt import generate_key; print(*generate_key())"
```

Run the notebook with those values:

```bash
uv run --env-file .env --with nbconvert --with ipywidgets jupyter nbconvert --execute --inplace src/preempt/demo.ipynb
```

## Key Branches

The following branches serve dedicated training and benchmarking workflows and are kept separate from `main`, each with its own `uv` dependency group in `pyproject.toml`:

| Branch | Dependency Group | Install Command | Purpose |
| --- | --- | --- | --- |
| `feat/training` | `train` | `uv sync --group train` | Training pipeline and scripts for fine-tuning the custom DistilBERT NER model for PII recognition (`NAME`, `AGE`, `MONEY`). |
| `feat/ner-bench` | `ner-bench` | `uv sync --group ner-bench` | Benchmarking and evaluation suite for the DistilBERT NER recognizer (detection accuracy, quantization, and latency). |
| `feat/pii-bench` | `pii-bench` | `uv sync --group pii-bench` | End-to-end downstream utility and privacy benchmark (`pii-bench/`), comparing LLM responses on original vs. sanitized prompts across $\epsilon$ values using verifiable tasks and LLM-as-a-judge. |

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

In order to exclude installation of packages from a specific group (e.g. docs),
run:

```bash
uv sync --no-group docs
```

`uv sync` installs the ONNX runtime. Running an exported ONNX file does not
need PyTorch or Optimum. The heavier recognizer and export stacks are extras:

```bash
uv sync --extra torch    # PyTorch DistilBERT backend
uv sync --extra gliner   # zero-shot GLiNER recognizer
uv sync --extra export   # export a checkpoint to ONNX
```

`DistilBertRecognizer` defaults to the PyTorch backend, so that constructor
needs the `torch` extra. Pass `backend="onnx"` to run an exported graph with
the default install.

If you're coming from `poetry` then you'll notice that the virtual environment
is actually stored in the project root folder and is by default named as `.venv`.
The other important note is that while `poetry` uses a "flat" layout of the project,
`uv` opts for the the "src" layout. (For more info, see [here](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/))

### Project Logs

#### `docs/LESSONS.md`

`docs/LESSONS.md` records knowledge that would otherwise be lost between sessions.

- Update it at the end of each development session and after every major bug fix.
- Record key decisions and the reasoning behind them, quirks in the code or its dependencies, and edge cases that caused problems or are easy to miss.
- Make each entry self-contained: state the symptom, the cause, and the fix or workaround.
- Keep entries short and specific. Update or remove entries that are no longer true.

#### `RELEASE_NOTES.md`

`RELEASE_NOTES.md` records user-facing changes for each release.

- Add an entry for every change that affects users. Internal refactors and test-only changes can be left out unless they change behavior.
- Group entries by version, newest first, and include the release date.
- Use the sections Added, Changed, Fixed, Removed, Deprecated, and Security.
- Describe the effect on users (what they can now do or what behaves differently), not the implementation.
- Call out breaking changes and any required migration steps explicitly.
