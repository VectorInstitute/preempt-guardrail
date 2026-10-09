"""Load the NER checkpoint and turn logits into exact character spans.

Accuracy and the PyTorch check use padded batches. Latency times one
unpadded string at a time, which is what a single request looks like.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preempt.bio import spans_from_tags
from training.prepare import normalize_span


# (start, end, label), the same tuple training/train.py scores.
Span = tuple[int, int, str]
Infer = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass
class Predictions:
    """Decoded test-set output for one model."""

    spans: list[list[Span]]
    tag_ids: list[list[int]]
    attention_mask: list[list[int]]
    truncated_entities: int
    dropped_spans: int


def load_tokenizer(model_dir: Path) -> Any:
    """Load the fast tokenizer saved beside the ONNX file.

    Parameters
    ----------
    model_dir : Path
        Directory that contains ``tokenizer.json``.

    Returns
    -------
    Any
        Hugging Face fast tokenizer.

    """
    return AutoTokenizer.from_pretrained(model_dir, use_fast=True)


def load_onnx_session(model_path: Path) -> ort.InferenceSession:
    """Open an ONNX file on CPU.

    CPU is the inference machine this benchmark is measuring. A GPU
    provider would time hardware the service may not have.

    Parameters
    ----------
    model_path : Path
        ``model.onnx`` or ``model.int8.onnx``.

    Returns
    -------
    onnxruntime.InferenceSession
        Session bound to ``CPUExecutionProvider``.

    """
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def load_pytorch(checkpoint: Path) -> torch.nn.Module:
    """Load the fine-tuned token classifier onto CPU.

    Parameters
    ----------
    checkpoint : Path
        Hugging Face checkpoint directory (``model.safetensors``).

    Returns
    -------
    torch.nn.Module
        Model in eval mode.

    """
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.to("cpu")
    model.eval()
    return model


def onnx_infer(session: ort.InferenceSession) -> Infer:
    """Return a function that feeds one encoded batch to ONNX Runtime.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Fp32 or INT8 session.

    Returns
    -------
    Infer
        Maps ``input_ids`` and ``attention_mask`` to logits.

    """

    def infer(encoded: dict[str, np.ndarray]) -> np.ndarray:
        feeds: dict[str, np.ndarray] = {}
        for item in session.get_inputs():
            if item.name not in encoded:
                raise KeyError(f"ONNX input {item.name} is missing from the tokenizer output")
            feeds[item.name] = encoded[item.name]
        return session.run(None, feeds)[0]

    return infer


def pytorch_infer(model: torch.nn.Module) -> Infer:
    """Return a function that runs one encoded batch through PyTorch.

    Parameters
    ----------
    model : torch.nn.Module
        Checkpoint loaded by :func:`load_pytorch`.

    Returns
    -------
    Infer
        Maps ``input_ids`` and ``attention_mask`` to logits.

    """

    def infer(encoded: dict[str, np.ndarray]) -> np.ndarray:
        # input_ids contains token IDs representing text
        # attention_mask indicates which tokens are real and which are padding
        # Convert NumPy arrays into PyTorch tensors - the model expects PyTorch tensors
        batch = {
            "input_ids": torch.from_numpy(encoded["input_ids"]),
            "attention_mask": torch.from_numpy(encoded["attention_mask"]),
        }
        with torch.no_grad():
            # .logits is the classification head: shape (batch, tokens, 15)
            # 15 is the tag set, not the 7 entity types. Each token gets a score for O plus \
            # B- and I- for NAME, AGE, MONEY, EMAIL, PHONE, ADDRESS, and ID.
            logits = model(**batch).logits
        return logits.numpy()

    return infer


def tags_and_scores(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Softmax, then argmax. This is the same reduction training uses.

    Parameters
    ----------
    logits : np.ndarray
        Shape ``(batch, sequence, labels)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tag ids and the max class probability, both ``(batch, sequence)``.

    """
    # Subtracting the maximum is a numerical-stability technique. \
    # It prevents exp() from overflowing.
    shifted = logits - logits.max(axis=-1, keepdims=True)
    # Exponentiate the shifted logits to get the probabilities.
    exp = np.exp(shifted)
    # Normalize the probabilities so they sum to 1.
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.argmax(axis=-1), probs.max(axis=-1)


def encode_batch(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    *,
    padding: bool,
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    """Tokenize a batch and keep character offsets for span decoding.

    Parameters
    ----------
    tokenizer : Any
        Fast tokenizer from :func:`load_tokenizer`.
    texts : list[str]
        Raw example strings.
    max_length : int
        Truncation length. Training uses 512.
    padding : bool
        True for the batched accuracy pass. Latency passes one string
        with padding off so the timer is not paying for pad tokens.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]
        Model inputs, and per-example ``(start, end)`` offsets.

    """

    # BatchEncoding is a dictionary-like object that contains the tokenized inputs, attention masks, and offset mappings.
    # The tokenizer.encode() method is used to tokenize the input texts.
    # The truncation=True argument ensures that the input texts are truncated to the specified max_length.
    # The padding=padding argument controls whether padding is added to the input texts to make them all the same length.
    # The return_offsets_mapping=True argument returns the character offsets for each token in the input texts.
    # The return_tensors="np" argument returns the tokenized inputs and attention masks as NumPy arrays.
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=padding,
        return_offsets_mapping=True,
        return_tensors="np",
    )
    offset_rows = [[(int(start), int(end)) for start, end in row] for row in encoded["offset_mapping"].tolist()]
    arrays = {
        "input_ids": np.asarray(encoded["input_ids"]),
        "attention_mask": np.asarray(encoded["attention_mask"]),
    }
    return arrays, offset_rows


def decode_example(
    text: str,
    offsets: list[tuple[int, int]],
    tag_ids: list[int],
    confidences: list[float],
    source: str,
) -> tuple[list[Span], int]:
    """Decode BIO tags and snap AGE and MONEY the way training does.

    Parameters
    ----------
    text : str
        Original example.
    offsets : list[tuple[int, int]]
        Character offsets parallel to ``tag_ids``.
    tag_ids : list[int]
        Argmax tag id per token.
    confidences : list[float]
        Max class probability per token.
    source : str
        Stored on each span (``fp32``, ``pytorch``, or ``int8``).

    Returns
    -------
    tuple[list[Span], int]
        Normalized spans, and how many tag pieces were dropped.

    """
    spans, dropped = spans_from_tags(text, offsets, tag_ids, confidences, source=source)
    normalized: list[Span] = []
    for span in spans:
        # Gold AGE is the digits and gold MONEY is the full amount.
        # Training applies this snap before exact-span F1.
        start, end = normalize_span(text, span.start, span.end, span.label)
        if start >= end:
            start, end = span.start, span.end
        normalized.append((start, end, span.label))
    return normalized, dropped


def _truncated_entities(
    tokenizer: Any,
    text: str,
    offsets: list[tuple[int, int]],
    row: dict[str, object],
    max_length: int,
) -> int:
    """Count gold entities cut off by ``max_length``.

    Same rule as ``training.train.predict_split``: only when the
    untruncated token count exceeds ``max_length``.
    """
    raw_entities = row["entities"]
    if not isinstance(raw_entities, list):
        raise TypeError("entities must be a list")
    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    if token_count <= max_length:
        return 0
    last = 0
    for start, end in offsets:
        if start != end and end > last:
            last = end
    return sum(1 for entity in raw_entities if int(entity["end"]) > last)


def predict(
    infer: Infer,
    tokenizer: Any,
    rows: list[dict[str, object]],
    *,
    max_length: int,
    batch_size: int,
    source: str,
) -> Predictions:
    """Run a model over prepared JSONL rows.

    Parameters
    ----------
    infer : Infer
        Fp32, INT8, or PyTorch forward pass.
    tokenizer : Any
        Fast tokenizer shared by every model, so the token ids match.
    rows : list[dict[str, object]]
        Records with ``text`` and ``entities``.
    max_length : int
        Truncation length.
    batch_size : int
        Examples per forward pass. Latency does not use this.
    source : str
        Name printed in progress lines and stored on spans.

    Returns
    -------
    Predictions
        Spans, tags, and the truncation counts training reports.

    """
    spans_out: list[list[Span]] = []
    tags_out: list[list[int]] = []
    mask_out: list[list[int]] = []
    truncated = 0
    dropped = 0
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch]
        arrays, offset_rows = encode_batch(tokenizer, texts, max_length, padding=True)
        tag_ids, scores = tags_and_scores(infer(arrays))
        mask = arrays["attention_mask"]
        for index, row in enumerate(batch):
            tags = [int(tag) for tag in tag_ids[index].tolist()]
            confidences = [float(score) for score in scores[index].tolist()]
            example_spans, example_dropped = decode_example(
                texts[index],
                offset_rows[index],
                tags,
                confidences,
                source,
            )
            dropped += example_dropped
            spans_out.append(example_spans)
            tags_out.append(tags)
            mask_out.append([int(flag) for flag in mask[index].tolist()])
            truncated += _truncated_entities(tokenizer, texts[index], offset_rows[index], row, max_length)
        done = min(start + batch_size, total)
        # A line every 50 batches, plus the last batch, so a 21k run is not silent.
        if start == 0 or (start // batch_size) % 50 == 0 or done == total:
            print(f"  {source}: {done}/{total}", flush=True)
    return Predictions(
        spans=spans_out,
        tag_ids=tags_out,
        attention_mask=mask_out,
        truncated_entities=truncated,
        dropped_spans=dropped,
    )


def run_example(infer: Infer, tokenizer: Any, text: str, max_length: int, source: str) -> None:
    """Tokenize, run, and decode a single string.

    The latency timer wraps this whole call. The average is the cost of
    one input, including tokenization and span decoding.

    Parameters
    ----------
    infer : Infer
        ONNX forward pass.
    tokenizer : Any
        Fast tokenizer.
    text : str
        One test example.
    max_length : int
        Truncation length.
    source : str
        Span source label. The timed call discards the spans.

    """
    arrays, offset_rows = encode_batch(tokenizer, [text], max_length, padding=False)
    tag_ids, scores = tags_and_scores(infer(arrays))
    decode_example(
        text,
        offset_rows[0],
        [int(tag) for tag in tag_ids[0].tolist()],
        [float(score) for score in scores[0].tolist()],
        source,
    )
