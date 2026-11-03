# Legend for the run tables

Each row of a run is one prompt, one ε, and one sanitizer draw. The generator is gpt-oss-120b. The judge is claude-sonnet-5-5.

## Answers

| Symbol | What it is |
|---|---|
| A | The generator's answer to the original prompt. |
| A′ | A second, independent answer to that same original prompt. Nothing was sanitized. This is the noise floor: how much the generator changes its mind when the prompt does not. |
| B | The generator's answer to the sanitized prompt, with surrogate names restored to the originals. Ages and money amounts stay noised. |

A vs A′ compares A with A′. A vs B compares A with B.

A larger ε adds less noise. Draw 0, 1, and 2 are three independent sanitizer seeds for the same prompt. A and A′ do not depend on ε or the draw, so that column repeats.

## Agreement (verifiable prompts only)

Verifiable prompts end with `FINAL_ANSWER:`. Agreement is how often those parsed answers match.

| Cell | Meaning |
|---|---|
| `33/36` | 33 of 36 parsed pairs matched. |
| Labels | Exact match, such as `ELIGIBLE` and `NOT_ELIGIBLE`. |
| Numbers | Match within 1% relative tolerance. |
| A missing count | The answer line did not parse. Those pairs are left out of the fraction. The parse-failure line counts them separately. |

Incidental prompts (a draft or a general explanation) have no `FINAL_ANSWER`. Their tables show judge scores only.

## Judge score

The judge sees the original prompt, the prompt the second answer was generated from, the substitution table, and the two answers. It scores the pair twice, with the answers swapped, so the first position is not favored. The table shows the mean of those two scores.

`overall_compatibility` is an integer from 1 to 5:

| Score | Meaning |
|---|---|
| 5 | Same decision and equally useful. |
| 4 | Equally well served, with only minor differences. |
| 3 | Mostly useful, but a noticeable gap. |
| 2 | A material omission or a shift in advice. The user is not equally well served. |
| 1 | A contradictory decision, or one answer fails the request. |

A different age or dollar figure is expected on B. The judge is told to penalize that difference only when it changes the decision, the advice, or whether the user is equally well served.

`4.62` in a judge column is that mean. It is the average over every pair in the cell (items × draws, and both presentation orders already collapsed into each pair's mean).

## Served

`equally_well_served` is the judge's yes/no: would these two answers serve the user equally well?

Each pair is asked twice. The report counts a pair as served only when both orders say yes.

| Cell | Meaning |
|---|---|
| `served 33/36` | Both orders said yes on 33 of 36 pairs. |
| The other 3 | Both orders said no, or the two orders disagreed (one yes, one no). A disagreement is stored as `mixed` and is not counted in the served numerator. |

Served is a coarser cut of the same judgment as the 1–5 score. A pair can score 4 and still be served. A pair that scores 1 or 2 is a no.
