# Run summary

Verifiable rows: 756. Incidental rows: 1512. Judge rows: 4534.
A vs A' is two samples of the original prompt. A vs B is the sanitized prompt with names restored.
Judge cells are the mean 1-5 score across both presentation orders, then how often both orders said the user was equally well served.
Verifiable prompts ask for a decision or a number, so a changed age or amount can change the answer.
Incidental prompts ask for a draft or a general explanation. The name, age, and amount are in the prompt, and the request does not turn on their exact values.

## Per epsilon (verifiable)

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.2 | 249/252 | 97/252 | 4.90, served 249/252 | 2.62, served 98/252 |
| 0.5 | 249/252 | 108/252 | 4.90, served 249/252 | 2.81, served 109/252 |
| 1 | 249/252 | 124/252 | 4.90, served 249/252 | 3.08, served 124/252 |

FINAL_ANSWER parse failures: 0/2268.

## Per family and epsilon

### age_gate

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.2 | 63/63 | 42/63 | 5.00, served 63/63 | 3.50, served 42/63 |
| 0.5 | 63/63 | 43/63 | 5.00, served 63/63 | 3.56, served 43/63 |
| 1 | 63/63 | 48/63 | 5.00, served 63/63 | 3.83, served 48/63 |

### raise_calculation

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.2 | 63/63 | 12/63 | 4.93, served 63/63 | 2.25, served 13/63 |
| 0.5 | 63/63 | 12/63 | 4.93, served 63/63 | 2.19, served 12/63 |
| 1 | 63/63 | 15/63 | 4.93, served 63/63 | 2.49, served 15/63 |

### retirement_eligibility

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.2 | 60/63 | 34/63 | 4.74, served 60/63 | 2.80, served 34/63 |
| 0.5 | 60/63 | 44/63 | 4.74, served 60/63 | 3.50, served 44/63 |
| 1 | 60/63 | 52/63 | 4.74, served 60/63 | 4.02, served 52/63 |

### savings_rate

| ε | A vs A′ | A vs B | Judge A vs A′ | Judge A vs B |
|---|---|---|---|---|
| 0.2 | 63/63 | 9/63 | 4.95, served 63/63 | 1.93, served 9/63 |
| 0.5 | 63/63 | 9/63 | 4.95, served 63/63 | 1.98, served 10/63 |
| 1 | 63/63 | 9/63 | 4.95, served 63/63 | 1.95, served 9/63 |

## Incidental prompts

No FINAL_ANSWER. The comparison is the judge against the A vs A' noise floor.

### email_drafting

| ε | Judge A vs A′ | Judge A vs B |
|---|---|---|
| 0.2 | 4.56, served 186/189 | 4.05, served 150/189 |
| 0.5 | 4.56, served 186/189 | 4.11, served 161/187 |
| 1 | 4.56, served 186/189 | 4.11, served 158/189 |

### general_request

| ε | Judge A vs A′ | Judge A vs B |
|---|---|---|
| 0.2 | 4.08, served 297/315 | 4.09, served 292/315 |
| 0.5 | 4.08, served 297/315 | 4.13, served 298/315 |
| 1 | 4.08, served 297/315 | 4.17, served 302/315 |
