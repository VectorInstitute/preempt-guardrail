"""PII recognition and sanitization."""

from preempt.contracts import EntitySpan, Recognizer
from preempt.legacy import LegacyNERShim
from preempt.sanitizer import Sanitizer, generate_key


# Recognizers are not re-exported: importing them loads transformers, or the
# optional gliner extra.


__all__ = ["EntitySpan", "LegacyNERShim", "Recognizer", "Sanitizer", "generate_key"]
