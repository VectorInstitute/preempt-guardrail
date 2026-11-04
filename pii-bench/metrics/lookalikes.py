"""See whether the recognizer tags strings that are not personal data.

These are look-alikes from the benchmark plan: a company named Apple, the
country Jordan, a bible verse, a version number, and similar lines. Nothing
here is written into the scored dataset. The printout is the evidence for
either adding a hard-negative stratum or recording the gap as a limitation.

Usage (from the ``pii-bench/`` directory)::

    uv run --project .. --group train python -m metrics.lookalikes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from preempt.distilbert_recognizer import DistilBertRecognizer


BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BENCH_DIR.parent

# Each line is ordinary text. A span here is a false hit.
CASES: tuple[tuple[str, str], ...] = (
    ("apple_company", "I bought an Apple laptop for the design team."),
    ("apple_fruit", "She packed an apple and a sandwich for lunch."),
    ("jordan_country", "They flew to Jordan for a week in the spring."),
    ("jordan_name", "Please ask Jordan to send the notes."),
    ("bill_noun", "The bill for dinner was split four ways."),
    ("bill_name", "Bill will join the call at noon."),
    ("mark_verse", "The reading in class was Mark 4:12."),
    ("mark_name", "Mark sent the revised draft yesterday."),
    ("version_number", "Install version 52 before Friday."),
    ("route", "We took Route 66 through the desert."),
    ("iphone", "The office ordered an iPhone 15 for the new hire."),
    ("universe_age", "The age of the universe is about 14 billion years."),
    ("rose_flower", "A rose bush grows beside the door."),
    ("will_modal", "She will file the form next week."),
    ("china_country", "The shipment from China arrived on Tuesday."),
    ("turkey_country", "The conference is in Turkey this year."),
    ("georgia_place", "The plant in Georgia ships on Mondays."),
    ("amazon_company", "Order the cables from Amazon."),
    ("dollar_idiom", "That advice is not worth a dollar."),
    ("early_retirement_phrase", "Early retirement is a common goal in the article."),
)


def main() -> None:
    """Print every look-alike the recognizer marks as NAME, AGE, or MONEY."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=REPO_DIR / "runs/407944/onnx")
    parser.add_argument("--onnx-file", default="model.onnx")
    args = parser.parse_args()

    recognizer = DistilBertRecognizer(
        model_path=str(args.model_path),
        backend="onnx",
        onnx_file_name=args.onnx_file,
    )
    hits = 0
    for name, text in CASES:
        spans = [span for span in recognizer.recognize(text) if span.label in {"NAME", "AGE", "MONEY"}]
        if not spans:
            print(f"{name}: none")
            continue
        hits += 1
        rendered = ", ".join(f"{span.label}:{span.text}" for span in spans)
        print(f"{name}: {rendered}")
    print(f"cases with a span: {hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
