# Release notes

## 0.1.0

First open-source release of the library. It finds names, ages, and money amounts in a prompt, replaces them, and can put the originals back after the model call.

### Install

```bash
uv sync
```

That install runs an exported ONNX graph. It does not install PyTorch.

```bash
uv sync --extra torch     # PyTorch backend for DistilBertRecognizer
uv sync --extra gliner    # GlinerRecognizer
uv sync --extra export    # export a checkpoint to ONNX
```

`DistilBertRecognizer` defaults to `backend="pytorch"`. Pass `backend="onnx"` when you only have the default install.

### Breaking change

`Sanitizer` no longer has a default key.

```python
from preempt import Sanitizer, generate_key

key, tweak = generate_key()
sanitizer = Sanitizer(recognizer, key, tweak)
```

The name permutation and the FF3 money cipher are both derived from `key` and `tweak`. Anyone who has the pair can undo FPE names and amounts. Keep the pair with your other secrets. `generate_key()` returns a 256-bit key and a 64-bit tweak, as hex.

A recognizer with a `recognize()` method can be passed straight in. `Sanitizer` wraps it. `LegacyNERShim` is still there if you want to call `extract()` yourself.

`from preempt import Sanitizer, generate_key, LegacyNERShim, EntitySpan` works. The recognizers stay on their own modules, because importing them loads `transformers` or the `gliner` extra.

### Demo

`src/preempt/demo.ipynb` reads three environment variables, listed in `.env.example`:

| Variable | Meaning |
| --- | --- |
| `PREEMPT_MODEL_PATH` | Hugging Face model id, or a local directory with an ONNX graph |
| `PREEMPT_KEY` | Hex FF3 key (32, 48, or 64 hex characters) |
| `PREEMPT_TWEAK` | Hex FF3 tweak (16 hex characters) |

The notebook loads `model.onnx`, the full-precision graph, not `model.int8.onnx`.

### Not in this release

- Training scripts stay on `feat/training`.
- NER evaluation stays on `feat/ner-bench`.
- The end-to-end privacy and utility benchmark stays on `feat/pii-bench`. This branch does not contain that code. The README table and the `pyproject.toml` dependency groups still name those branches.
- `training-data/` is gitignored. A local copy of the PII-masking train set can live there (`pii_masking_200k_en_fr_de_train.json`, 110,790 labeled conversations, about 85 MB). No code in this branch reads it, and it is not part of the package.

### Known limitations

- Replacement is a plain substring. A short value can match inside a longer word, and some decrypt paths restore only the first match.
- FPE money drops thousands separators. `$10,000` comes back as `$10000`.
- Ages, and money when you pass `use_fpe=False`, are noised with m-LDP. Restoring them needs the same `Sanitizer` instance that encrypted them. A new instance with the same key is not enough.
- Names that are not in the fixed lists also need that saved state. `decrypt` restores only surrogates from the last `encrypt` on that instance. Pass `key_only=True` to invert listed names and FPE amounts with the key alone. Key-only money skips an amount FF3 rejects instead of failing the batch. Ages have no key-only inverse.
- A value that cannot be encrypted is replaced with `[REDACTED]`. Its row index is still returned. Other values in that text are still replaced.
- DistilBERT and GLiNER read text longer than the model window in overlapping pieces. An entity longer than the overlap (`stride` tokens for DistilBERT, `overlap` words for GLiNER) can still be split.
- The sanitizer only replaces `Name`, `Money`, and `Age`. The recognizer can also emit email, phone, address, and id spans, and those are left as they are.
