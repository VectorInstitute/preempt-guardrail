"""DistilBERT token classifier that returns character spans."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer

from preempt.bio import ID_TO_TAG, spans_from_tags
from preempt.contracts import EntitySpan
from preempt.spans import currency_spans, normalize_span


class DistilBertRecognizer:
    """Load a fine-tuned DistilBERT checkpoint and emit character spans.

    The ONNX backend runs an exported graph with onnxruntime. The PyTorch
    backend needs the ``torch`` extra.
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 512,
        backend: Literal["pytorch", "onnx"] = "pytorch",
        onnx_file_name: str | None = None,
    ) -> None:
        """Load the tokenizer and token-classification head.

        Parameters
        ----------
        model_path : str
            Hugging Face model id or a local checkpoint directory.
        max_length : int
            Maximum tokens per call. Longer text is truncated.
        backend : Literal["pytorch", "onnx"]
            Backend to use for inference. ``onnx`` reads an exported graph
            and does not import PyTorch.
        onnx_file_name : str | None
            ONNX file to load from ``model_path``, such as ``model.onnx`` or
            ``model.int8.onnx``. Required only when the directory contains
            multiple ONNX model variants.

        Raises
        ------
        ImportError
            ``backend`` is ``pytorch`` and the ``torch`` extra is not installed.

        """
        if backend not in ("pytorch", "onnx"):
            raise ValueError(f"unknown backend: {backend}")
        self.backend: Literal["pytorch", "onnx"] = backend
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_path)
        if backend == "pytorch":
            if onnx_file_name is not None:
                raise ValueError("onnx_file_name can only be used with the ONNX backend")
            torch = _require_torch()
            from transformers import AutoModelForTokenClassification  # noqa: PLC0415

            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            self._torch = torch
            self.session = None
        else:
            onnx_file = _onnx_file(model_path, onnx_file_name)
            self.session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
            self.model = None
            self._torch = None
        self._check_labels()

    def _check_labels(self) -> None:
        raw = self.config.id2label
        actual = {int(key): str(value) for key, value in raw.items()}
        if actual != ID_TO_TAG:
            raise ValueError("checkpoint labels do not match the BIO label set")

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return PII spans in ``text``.

        Parameters
        ----------
        text : str
            Source text.

        Returns
        -------
        list[EntitySpan]
            Spans sliced from ``text``. ``source`` is ``distilbert``.

        """
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        # pop offset_mapping from the encoded dictionary because model doesn't accept it
        # [0] selects first batch item.
        # offset_mapping is [batch_size, tokens, 2], where the last axis is
        # the character start and end.
        offset_rows = encoded.pop("offset_mapping")[0].tolist()
        offsets = [(int(start), int(end)) for start, end in offset_rows]
        logits = self._onnx_logits(encoded) if self.backend == "onnx" else self._pytorch_logits(encoded)
        # probs is [tokens, num_labels]
        probs = _softmax(logits)
        # tag_ids is [tokens]
        tag_ids = [int(tag_id) for tag_id in probs.argmax(axis=-1).tolist()]
        # confidences is [tokens]
        confidences = [float(score) for score in probs.max(axis=-1).tolist()]
        spans, _dropped = spans_from_tags(text, offsets, tag_ids, confidences, source="distilbert")
        # Snap AGE to its digits and MONEY to the full amount, same as training eval.
        # NAME and the other labels keep the decoded bounds.
        normalized: list[EntitySpan] = []
        for span in spans:
            start, end = normalize_span(text, span.start, span.end, span.label)
            if start >= end:
                start, end = span.start, span.end
            normalized.append(
                EntitySpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=span.label,
                    confidence=span.confidence,
                    source=span.source,
                )
            )
        # The checkpoint tags many "$10,000" style amounts as O, or as an ID
        # that covers only the digits before the comma. A symbol-led amount
        # fills that gap unless a MONEY span already covers it.
        for start, end in currency_spans(text):
            if any(span.label == "MONEY" and span.start <= start and span.end >= end for span in normalized):
                continue
            normalized = [
                span for span in normalized if not (span.label == "MONEY" and span.start < end and start < span.end)
            ]
            normalized.append(
                EntitySpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label="MONEY",
                    confidence=1.0,
                    source="currency",
                )
            )
        normalized.sort(key=lambda span: (span.start, span.end))
        return normalized

    def _onnx_logits(self, encoded: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("ONNX session is not loaded")
        feeds: dict[str, np.ndarray] = {}
        for item in self.session.get_inputs():
            if item.name not in encoded:
                raise ValueError(f"tokenizer did not produce ONNX input {item.name}")
            value = encoded[item.name]
            if value.dtype.kind in "iu":
                value = value.astype(np.int64)
            feeds[item.name] = value
        outputs = self.session.run(["logits"], feeds)
        # logits is [batch_size, tokens, num_tags]. [0] drops the batch axis.
        return np.asarray(outputs[0][0])

    def _pytorch_logits(self, encoded: Mapping[str, np.ndarray]) -> np.ndarray:
        torch = self._torch if self._torch is not None else _require_torch()
        if self.model is None:
            raise RuntimeError("PyTorch model is not loaded")
        batch = {name: torch.from_numpy(value) for name, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**batch).logits[0]
        return logits.detach().cpu().numpy()


def _require_torch() -> Any:
    """Import torch, or raise with the extra that provides it."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("the PyTorch backend requires the torch extra: uv sync --extra torch") from exc
    return torch


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _onnx_file(model_path: str, onnx_file_name: str | None) -> str:
    path = Path(model_path)
    if path.is_dir():
        if onnx_file_name is None:
            matches = sorted(path.glob("*.onnx"))
            if len(matches) == 1:
                return str(matches[0])
            if not matches:
                raise FileNotFoundError(f"no ONNX file found in {path}")
            raise ValueError("onnx_file_name is required when the directory contains multiple ONNX files")
        candidate = path / onnx_file_name
        if not candidate.is_file():
            raise FileNotFoundError(f"ONNX file not found: {candidate}")
        return str(candidate)
    if path.is_file():
        raise ValueError("model_path must be a directory or a Hugging Face model id")
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return hf_hub_download(repo_id=model_path, filename=onnx_file_name or "model.onnx")
