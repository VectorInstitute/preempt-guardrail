# Epsilon comparison (retirement eligibility)

Name-fixed sanitizer. 12 retirement items, draws 0–2, so 36 rows per ε.
Generator: gpt-oss-120b. Judge: claude-sonnet-5-5, both presentation orders.
A is the answer to the original prompt, A′ is a second sample of that same prompt, and B is the answer to the sanitized prompt with names restored. Ages and money stay noised.

Source: `pii-bench/runs/sweep/` (`results.jsonl`, `judge.jsonl`).

A larger ε adds less noise. Agreement rising with ε is the expected direction.

## Pooled across draws


| ε   | A vs A′ | A vs B | Judge A vs A′      | Judge A vs B       |
| --- | ------- | ------ | ------------------ | ------------------ |
| 0.1 | 33/36   | 25/36  | 4.62, served 33/36 | 3.42, served 25/36 |
| 0.2 | 33/36   | 32/36  | 4.62, served 33/36 | 4.25, served 32/36 |
| 0.5 | 33/36   | 36/36  | 4.62, served 33/36 | 4.64, served 36/36 |
| 1   | 33/36   | 36/36  | 4.62, served 33/36 | 4.86, served 36/36 |


Judge columns are the mean of the two orderings (`overall_compatibility`, 1–5) and how often both orderings said the user was equally well served. No ordering pair was mixed. No `FINAL_ANSWER` failed to parse.

## By draw

A vs A′ is 11/12 on every cell. A vs B:


| ε   | Draw 0 | Draw 1 | Draw 2 |
| --- | ------ | ------ | ------ |
| 0.1 | 10/12  | 6/12   | 9/12   |
| 0.2 | 11/12  | 9/12   | 12/12  |
| 0.5 | 12/12  | 12/12  | 12/12  |
| 1   | 12/12  | 12/12  | 12/12  |




## What the numbers mean

A vs A′ does not depend on ε or the draw.  The repeated miss is `retirement_eligibility-001-00`, the old-401(k) tThose two answers are samples of the original prompt, so the column is the same 33/36 everywhere.emplate whose rule of 55 makes the label ambiguous. Both samples saw the original age.

Every A vs B miss is an age noised across 59½, which flips `ELIGIBLE` / `NOT_ELIGIBLE`. That happens on 11 of 36 rows at ε=0.1, 4 of 36 at ε=0.2, and on none at ε=0.5 or ε=1. The four ε=0.2 misses are draw 0 `retirement_eligibility-000-03` (46→60) and draw 1 `000-01`, `001-01`, and `001-03`.

The judge's yes/no count matches those label flips exactly at every ε. Mean compatibility for A vs B moves from 3.42 at ε=0.1 to 4.86 at ε=1. The A vs A′ judge baseline is 4.62. At ε=0.5 the sanitized pair is level with that baseline (4.64, served 36/36). At ε=1 it is higher (4.86) because every sanitized pair agrees, while the baseline still includes the one contradictory resample.

## Other runs, not in the table

`runs/pilot-e0.2/` is the same ε=0.2 retirement rows (draws 0–2) plus 12 email drafts at draw 0 only. Its judge file scores draw 0 only, both families: A vs A′ mean 4.50 (served 23/24), A vs B mean 4.29 (served 20/24, 3 mixed). The mixed rows are email drafts where one ordering treated the noised dollar amount as a wrong figure in the email and the other treated it as expected noise.

Runs from before the name-list fix (`runs/eps0.1`, `runs/eps0.2`, `runs/pilot`) are draw 0 only and are not comparable to this table.