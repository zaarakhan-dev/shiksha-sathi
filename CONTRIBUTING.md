# Contributing — Team Disha

## Branches
- `main` — demo-ready at all times. Never commit directly.
- `feat/<area>-<short-desc>` — e.g. `feat/ml-confidence-scoring`
- Open a PR into `main`; one teammate reviews before merge.

## Ownership
| Area | Path | Owner |
|---|---|---|
| Diagnostic engine | `ml/` | Zaara |
| API | `backend/` | Bhupender |
| UI | `frontend/` | Alok |
| Competency taxonomy | `data/`, `docs/` | Aditi |
| Evaluation & research | `docs/` | Dikshita |
| Demo script & QA | — | Nikhil |

## Before you push
```bash
python -m pytest tests/ -v
```
All tests must pass. If you change scoring or diagnosis behaviour, add a test
named after the question it answers.

## Rules
1. No hardcoded demo values in the UI — read from the API.
2. Any new claim in the README must be backed by code or marked as roadmap.
3. Never commit real personal data. Synthetic only.
