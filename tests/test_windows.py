"""Windows for text that is longer than one model call."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from preempt.contracts import EntitySpan
from preempt.distilbert_recognizer import token_windows
from preempt.spans import covering_windows, merge_overlapping_spans


def _tokenizer() -> PreTrainedTokenizerFast:
    """Whitespace tokenizer that keeps an offset for every word in the tests."""
    words = ["Alice", "Bob", "filler", "is", "here"]
    vocab = {word: index + 4 for index, word in enumerate(words)}
    vocab.update({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


def test_covering_windows_reaches_the_last_piece() -> None:
    """Every piece sits inside a slice, and the last slice reaches the text end."""
    pieces = [(index * 2, index * 2 + 1) for index in range(10)]
    windows = covering_windows(pieces, size=4, overlap=2, text_length=20)

    assert windows[0][0] == 0
    assert windows[-1][1] == 20
    for start, end in pieces:
        assert any(window_start <= start and end <= window_end for window_start, window_end in windows)


def test_short_text_is_one_window() -> None:
    """A text that fits is not split."""
    text = "Alice Bob"
    assert token_windows(_tokenizer(), text, max_length=32, stride=4) == [(0, len(text))]


def test_token_windows_include_a_name_past_the_first_window() -> None:
    """The tail of a long text is inside a window, not dropped."""
    text = "Alice " + "filler " * 30 + "Bob is here"
    windows = token_windows(_tokenizer(), text, max_length=12, stride=4)
    bob = text.rindex("Bob")

    assert windows[0][0] == 0
    assert windows[-1][1] == len(text)
    assert any(start <= bob < end for start, end in windows)
    assert len(windows) > 1


def test_merge_keeps_the_longer_span_and_a_later_one() -> None:
    """A partial span loses to the full one, and a later span stays."""
    partial = EntitySpan(0, 5, "Alice", "NAME", 0.4, "distilbert")
    full = EntitySpan(0, 11, "Alice Smith", "NAME", 0.9, "distilbert")
    later = EntitySpan(20, 23, "Bob", "NAME", 0.8, "distilbert")

    merged = merge_overlapping_spans([partial, later, full])

    assert merged == [full, later]
