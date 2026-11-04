# Limitations and NER failure (360 prompts)

The scored set is 36 templates × 10 fills = 360 prompts in `data/dataset.jsonl`.
240 are incidental (email drafts and general explanations). 120 are verifiable
(retirement, age gates, savings rate, raise). The recognizer and sanitizer
were run on all 360 (`metrics.detection`, ε = 1). The generator and judge
have not been run on this larger set. `runs/sweep-v2/` is still the earlier
90-prompt sweep.

## Formats we do not generate

The recognizer and sanitizer do not handle these, so they are not in the
dataset. They stay limitations, not experiments.

- Ages written as words, birth dates, birth years, decades ("early 50s"), or
  relative ages ("I'll be 60 next month").
- Money written as words ("eighty-five thousand"), slang ("about 85 grand"),
  ranges ("between 80 and 90k"), or relations ("2x my spouse's salary").

Generated ages are integers. Generated money is one of `$12,000`, `$12k`, or
`12,000 dollars`.

## NER failure on the formats we do generate

Partial recall (IoU ≥ 0.5) against the gold spans. A miss is a gold span the
recognizer did not cover. Full per-item rows are in
`runs/detection_validation.jsonl`.

| Label | Format | Found | Failure rate |
|---|---|---|---|
| AGE | integer | 201/210 | 0.04 |
| MONEY | `$12,000` | 107/107 | 0.00 |
| MONEY | `$12k` | 134/134 | 0.00 |
| MONEY | `12,000 dollars` | 15/119 | 0.87 |
| NAME | full name | 408/470 | 0.13 |

`N dollars` is the format that fails. Comma amounts and `k` amounts are
found. After sanitization, an original value still appears in 145/360 prompts
(0.40), mostly money (117/360) and names (64/470).

Strict money F1 is 0.40 because the recognizer labels the digits inside
`$12,000` while the gold span includes the dollar sign. Partial F1, which
allows that overlap, is 0.83.

## Look-alikes

`metrics.lookalikes` ran the recognizer on 20 lines that are not personal
records (a company, a country, a verse, a version number, and similar).
3 of 20 lines produced a span:

| Line | Span |
|---|---|
| "They flew to Jordan for a week in the spring." | NAME: Jordan |
| "Please ask Jordan to send the notes." | NAME: Jordan |
| "Bill will join the call at noon." | NAME: Bill |

Apple, "the bill for dinner", Mark 4:12, version 52, Route 66, iPhone 15,
"the age of the universe", China, Turkey, and Georgia were not tagged.

This round does not add a hard-negative stratum. The gap worth a later slice
is place names that are also given names. Ordinary look-alikes in the list
above were left alone.

## What is not in this file

Privacy attacks, human labels for the judge, bootstrap intervals, and
per-type ablations are later. A full ε sweep of these 360 prompts is a
separate generator and judge run.
