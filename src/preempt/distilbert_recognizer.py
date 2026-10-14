"""DistilBERT token classifier that returns character spans."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from preempt.bio import ID_TO_TAG, spans_from_tags
from preempt.contracts import EntitySpan
from preempt.spans import currency_spans, normalize_span


class DistilBertRecognizer:
    """Load a fine-tuned DistilBERT checkpoint and emit character spans."""

    def __init__(self, model_path: str, max_length: int = 512) -> None:
        """Load the tokenizer and token-classification head.

        Parameters
        ----------
        model_path : str
            Hugging Face model id or a local checkpoint directory.
        max_length : int
            Maximum tokens per call. Longer text is truncated.

        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self._check_labels()
        self.model.eval()

    def _check_labels(self) -> None:
        raw = self.model.config.id2label
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
            return_tensors="pt",
        )
        # pop offset_mapping from the encoded dictionary because model doesn't accept it
        # [0] selects first batch item.
        # offset_mapping is [batch_size, tokens, 2], where the last axis is
        # the character start and end.
        offset_rows = encoded.pop("offset_mapping")[0].tolist()
        offsets = [(int(start), int(end)) for start, end in offset_rows]
        with torch.no_grad():
            # logits is [batch_size, tokens, num_tags]
            # logits[0] drops the batch axis, leaving [tokens, num_labels]
            logits = self.model(**encoded).logits[0]
            # probs is [tokens, num_labels]
            probs = torch.softmax(logits, dim=-1)
            # tag_ids is [tokens]
            tag_ids = [int(tag_id) for tag_id in probs.argmax(dim=-1).tolist()]
            # confidences is [tokens]
            confidences = [float(score) for score in probs.max(dim=-1).values.tolist()]
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
            if any(
                span.label == "MONEY" and span.start <= start and span.end >= end
                for span in normalized
            ):
                continue
            normalized = [
                span
                for span in normalized
                if not (span.label == "MONEY" and span.start < end and start < span.end)
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
