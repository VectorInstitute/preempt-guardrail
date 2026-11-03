# Run summary

Verifiable rows: 720. Incidental rows: 360. Judge rows: 2160.
A vs A' is two samples of the original prompt. A vs B is the sanitized prompt with names restored.
Judge cells are the mean 1-5 score across both presentation orders, then how often both orders said the user was equally well served.
Verifiable prompts ask for a decision or a number, so a changed age or amount can change the answer.
Incidental prompts ask for a draft or a general explanation. The name, age, and amount are in the prompt, and the request does not turn on their exact values.

## Per epsilon (verifiable)

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.1 | 177/180 | 72/180 | 4.91, served 177/180 | 2.55, served 71/180 |
| 0.2 | 177/180 | 71/180 | 4.91, served 177/180 | 2.63, served 72/180 |
| 0.5 | 177/180 | 78/180 | 4.91, served 177/180 | 2.77, served 78/180 |
| 1 | 177/180 | 93/180 | 4.91, served 177/180 | 3.13, served 93/180 |

FINAL_ANSWER parse failures: 0/2160.

## Per family and epsilon

### age_gate

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.1 | 45/45 | 27/45 | 5.00, served 45/45 | 3.18, served 27/45 |
| 0.2 | 45/45 | 27/45 | 5.00, served 45/45 | 3.22, served 27/45 |
| 0.5 | 45/45 | 27/45 | 5.00, served 45/45 | 3.23, served 27/45 |
| 1 | 45/45 | 32/45 | 5.00, served 45/45 | 3.63, served 32/45 |

### raise_calculation

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.1 | 45/45 | 10/45 | 5.00, served 45/45 | 2.12, served 9/45 |
| 0.2 | 45/45 | 9/45 | 5.00, served 45/45 | 2.34, served 10/45 |
| 0.5 | 45/45 | 9/45 | 5.00, served 45/45 | 2.21, served 9/45 |
| 1 | 45/45 | 12/45 | 5.00, served 45/45 | 2.60, served 12/45 |

### retirement_eligibility

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.1 | 42/45 | 26/45 | 4.63, served 42/45 | 2.93, served 26/45 |
| 0.2 | 42/45 | 26/45 | 4.63, served 42/45 | 2.91, served 26/45 |
| 0.5 | 42/45 | 33/45 | 4.63, served 42/45 | 3.60, served 33/45 |
| 1 | 42/45 | 40/45 | 4.63, served 42/45 | 4.23, served 40/45 |

### savings_rate

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.1 | 45/45 | 9/45 | 5.00, served 45/45 | 1.96, served 9/45 |
| 0.2 | 45/45 | 9/45 | 5.00, served 45/45 | 2.06, served 9/45 |
| 0.5 | 45/45 | 9/45 | 5.00, served 45/45 | 2.06, served 9/45 |
| 1 | 45/45 | 9/45 | 5.00, served 45/45 | 2.06, served 9/45 |

## Incidental prompts

No FINAL_ANSWER. The comparison is the judge against the A vs A' noise floor.

### email_drafting

| ε | Judge A vs A′ | Judge A vs B |
|---|---|---|
| 0.1 | 4.37, served 45/45 | 4.07, served 37/45 |
| 0.2 | 4.37, served 45/45 | 4.02, served 34/45 |
| 0.5 | 4.37, served 45/45 | 4.11, served 38/45 |
| 1 | 4.37, served 45/45 | 4.16, served 38/45 |

### general_request

| ε | Judge A vs A′ | Judge A vs B |
|---|---|---|
| 0.1 | 4.20, served 45/45 | 4.23, served 44/45 |
| 0.2 | 4.20, served 45/45 | 4.20, served 45/45 |
| 0.5 | 4.20, served 45/45 | 4.26, served 45/45 |
| 1 | 4.20, served 45/45 | 4.37, served 45/45 |
