"""Adapt a span recognizer to the extract() shape Sanitizer already calls."""

from preempt.contracts import SANITIZER_ENTITY, Recognizer
from preempt.spans import join_name_spans


class LegacyNERShim:
    """Presents a Recognizer as ``NER.extract`` for Name, Money, and Age."""

    def __init__(self, recognizer: Recognizer) -> None:
        """Store the recognizer used for extract().

        Parameters
        ----------
        recognizer : Recognizer
            Span recognizer to adapt.

        """
        self.recognizer = recognizer

    def extract(
        self,
        prompts: list[str],
        entity_type: str,
        batch_size: int = 1,
        output_dict: dict[str, list[list[str]]] | None = None,
    ) -> dict[str, list[list[str]]]:
        """Return surface strings for one sanitizer entity type.

        Parameters
        ----------
        prompts : list[str]
            Input texts, one string per row.
        entity_type : str
            ``Name``, ``Money``, or ``Age``.
        batch_size : int
            Unused. Accepted so the call matches ``NER.extract``.
        output_dict : dict[str, list[list[str]]] | None
            Existing dictionary to update.

        Returns
        -------
        dict[str, list[list[str]]]
            ``{entity_type: [[span text, ...], ...]}``, one list per prompt.
            Adjacent ``Name`` spans separated only by whitespace are one string.

        """
        del batch_size
        try:
            label = SANITIZER_ENTITY[entity_type]
        except KeyError:
            raise ValueError(f"unsupported entity type: {entity_type}") from None
        if output_dict is None:
            output_dict = {}
        result = []
        for prompt in prompts:
            spans = [span for span in self.recognizer.recognize(prompt) if span.label == label]
            extracted_texts = [span.text for span in spans]
            if entity_type == "Name":
                extracted_texts = join_name_spans(prompt, spans)
            result.append(extracted_texts)
        output_dict[entity_type] = result

        return output_dict
