"""Zero-shot GLiNER recognizer that returns character spans."""

from collections.abc import Mapping

from gliner import GLiNER

from preempt.contracts import LABELS, EntitySpan


# The quick-start zero-shot checkpoint. Swap this for a fine-tune later.
DEFAULT_MODEL_ID = "gliner-community/gliner_small-v2.5"

# Pipeline label -> prompt the model scores. Short type names match the
# labels GLiNER saw in pretraining; recognize() still emits the pipeline label.
LABEL_PROMPTS: dict[str, str] = {
    "NAME": "person",
    "AGE": "age",
    "MONEY": "money amount",
    "EMAIL": "email address",
    "PHONE": "phone number",
    "ADDRESS": "address",
    "ID": "identification number",
}


class GlinerRecognizer:
    """Load a zero-shot GLiNER checkpoint and emit character spans."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_ID,
        threshold: float = 0.5,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Load the model and the label prompts it will score.

        Parameters
        ----------
        model_path : str
            Hugging Face model id or a local checkpoint directory.
        threshold : float
            Minimum score for a span to be kept.
        labels : Mapping[str, str] | None
            Pipeline label to prompt text. Defaults to ``LABEL_PROMPTS``.
            Prompt strings must be unique. Returned spans use the keys.

        """
        self.threshold = threshold
        self.labels = _check_labels(LABEL_PROMPTS if labels is None else labels)
        self.model = GLiNER.from_pretrained(model_path)
        self.model.eval()

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return PII spans in ``text``.

        Parameters
        ----------
        text : str
            Source text.

        Returns
        -------
        list[EntitySpan]
            Spans sliced from ``text``. ``source`` is ``gliner``.
            Overlapping spans are dropped (flat NER).

        """
        if not text:
            return []
        entities = self.model.predict_entities(
            text,
            self.labels,
            flat_ner=True,
            threshold=self.threshold,
        )
        spans: list[EntitySpan] = []
        for entity in entities:
            label = str(entity["label"])
            if label not in self.labels:
                raise ValueError(f"unknown label: {label}")
            start = int(entity["start"])
            end = int(entity["end"])
            if start < 0 or end > len(text) or start >= end:
                continue
            spans.append(
                EntitySpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=label,
                    confidence=float(entity["score"]),
                    source="gliner",
                )
            )
        spans.sort(key=lambda span: (span.start, span.end))
        return spans


def _check_labels(labels: Mapping[str, str]) -> dict[str, str]:
    checked = dict(labels)
    prompts = list(checked.values())
    if len(set(prompts)) != len(prompts):
        raise ValueError("label prompts must be unique")
    for label in checked:
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
    return checked
