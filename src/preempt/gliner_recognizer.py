"""Zero-shot GLiNER recognizer that returns character spans."""

from collections.abc import Mapping

from preempt.contracts import LABELS, EntitySpan
from preempt.spans import covering_windows, merge_overlapping_spans


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
        overlap: int = 32,
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
        overlap : int
            Words repeated from the previous window when the text is longer
            than the model's ``max_len``. An entity longer than this can still
            be split.

        Raises
        ------
        ImportError
            The ``gliner`` extra is not installed.
        ValueError
            ``overlap`` is negative.

        """
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        self.threshold = threshold
        self.overlap = overlap
        self.labels = _check_labels(LABEL_PROMPTS if labels is None else labels)
        # Imported here so an install without the gliner extra does not need it.
        try:
            from gliner import GLiNER  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("GlinerRecognizer requires the gliner extra: uv sync --extra gliner") from exc
        self.model = GLiNER.from_pretrained(model_path)
        self.model.eval()

    def recognize(self, text: str) -> list[EntitySpan]:
        """Return PII spans in ``text``.

        Text longer than the model's ``max_len`` is read in overlapping word
        windows. GLiNER would otherwise truncate and leave the tail unscanned.

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
        windows = self._windows(text)
        spans: list[EntitySpan] = []
        for char_start, char_end in windows:
            spans.extend(self._spans_in_window(text, char_start, char_end))
        if len(windows) > 1:
            spans = merge_overlapping_spans(spans)
        spans.sort(key=lambda span: (span.start, span.end))
        return spans

    def _windows(self, text: str) -> list[tuple[int, int]]:
        """Character slices that stay within the model's word limit."""
        max_words = int(getattr(self.model.config, "max_len", 0) or 0)
        processor = getattr(self.model, "data_processor", None)
        splitter = getattr(processor, "words_splitter", None)
        if max_words < 1 or splitter is None:
            return [(0, len(text))]
        words = [(int(start), int(end)) for _token, start, end in splitter(text) if int(start) != int(end)]
        overlap = min(self.overlap, max_words - 1)
        return covering_windows(words, size=max_words, overlap=overlap, text_length=len(text))

    def _spans_in_window(self, text: str, char_start: int, char_end: int) -> list[EntitySpan]:
        """Score one slice and return spans in the coordinates of ``text``."""
        chunk = text[char_start:char_end]
        if not chunk:
            return []
        entities = self.model.predict_entities(
            chunk,
            self.labels,
            flat_ner=True,
            threshold=self.threshold,
        )
        spans: list[EntitySpan] = []
        for entity in entities:
            label = str(entity["label"])
            if label not in self.labels:
                raise ValueError(f"unknown label: {label}")
            start = int(entity["start"]) + char_start
            end = int(entity["end"]) + char_start
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
