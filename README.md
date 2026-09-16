# Shiksha Sathi (शिक्षा साथी)

### Competency Intelligence Layer for Capacity Building in India's Official Statistical System

**Smart India Hackathon 2026** · Problem Statement **SIH26101** (MoSPI) · Theme: Smart Education · Category: Software

> **From Skill Gaps to Personalized Learning**
>
> Shiksha Sathi does not deliver courses. It answers the question that comes *before* a course: **which competency is actually missing, why, what must be learned first, and did the gap actually close?**

---

## The problem

Officials in the statistical system have access to large learning catalogues such as iGOT Karmayogi. The gap is not content availability — it is decision support:

| Observed problem | Consequence |
|---|---|
| Training is tracked by modules completed | Completion is recorded, competency is not verified |
| A weak advanced skill is treated as the problem | The real blocker may be an untested foundation further upstream |
| Assessment scores and appraisal ratings live in separate systems | Disagreements between them are never detected |
| No cohort-level view | A trainer cannot see that most of a batch is stuck on the same foundation |

---

## What this system does

```
Phase 1  Trainer sets up a cohort and uploads a roster
Phase 2  Assessment  (self-serve link  OR  trainer-entered offline scores)
Phase 3  Scoring     composite quiz + appraisal, with conflict detection
Phase 3b Diagnosis   prerequisite DAG traversal to the deepest weak foundation
Phase 4  Remediation ordered learning path + AI-drafted MCQs (trainer approves)
Phase 5  Re-assess   measure before vs after
Phase 6  Dashboard   cohort heatmap, shared bottlenecks, per-learner drill-down
Phase 7  (roadmap)   verified competency growth feeds institutional processes
```

### The core idea, concretely

An officer scores **38% in Data Visualization**. A conventional recommender suggests a visualization course.

Shiksha Sathi walks the prerequisite graph first:

```
Python_Basics(80) ─┐
                   ├─► Data_Handling(42) ──► Pandas_Manipulation(40) ──► Data_Visualization(38)
SQL_Basics(76)  ───┘
```

The diagnosis is that **Data Handling** is the blocker. Remediating it is expected to unblock two downstream competencies. The learning path is ordered by prerequisite, not by score.

---

## What makes this defensible, not just clever

These are deliberate engineering choices in response to the ways this kind of system usually fails.

**1. Unassessed ≠ mastered.**
The competency graph contains 11 competencies. A typical assessment covers fewer. Untested competencies are reported as `unassessed` and **lower the confidence of any diagnosis that depends on them**. They are never silently treated as strong. Every response includes a coverage report.

**2. A conflicted score is never presented as a reliable number.**
If the quiz and appraisal signals diverge by ≥30 points, the composite is still computed (a trainer needs something actionable) but it is flagged, confidence drops to `low`, and **both raw signals stay visible** so nobody mistakes the average for a measurement.

**3. Weights are policy, not physics.**
The 70/30 split is a configurable default for the pilot, validated at construction time. A deploying ministry would calibrate it against outcome data.

**4. Every diagnosis carries an evidence trail.**
No conclusion is asserted without the reasoning that produced it — the score, the threshold, the prerequisite path walked, and any gaps in the evidence.

**5. AI output is gated by a human.**
Question generation is assistive. Nothing generated reaches a learner without explicit trainer approval, and rejections return for regeneration.

---

## Scoring and confidence

| Situation | Mode | Confidence |
|---|---|---|
| Quiz + appraisal agree | `composite` | high |
| Quiz only (cold start) | `quiz_only` | medium |
| Quiz + appraisal diverge ≥30 pts | `composite_conflicted` | low + flagged |
| Appraisal only, no objective test | `appraisal_only` | low |
| No data | `unassessed` | excluded from diagnosis |

A diagnosis is additionally downgraded to `low` if **any** prerequisite in its chain was never assessed.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Diagnostic engine | Python, NetworkX | DAG ancestor traversal is the natural fit for prerequisites |
| API | FastAPI, Pydantic | Typed request validation, auto-generated OpenAPI docs |
| Frontend | HTML + Tailwind (prototype) | Fast to iterate for a demo; React is the production path |
| Data | In-memory store, MongoDB-shaped | Route signatures unchanged when swapping in a real DB |
| Question generation | Self-hosted local LLM | **Data never leaves the deployment** — relevant for government use |

---

## Running it

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs

# Frontend (separate terminal)
cd frontend && python -m http.server 3000
# http://localhost:3000
```

### Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

14 tests, each named for the evaluation question it answers — including
`test_unassessed_competency_is_not_assumed_mastered` and
`test_conflicting_signals_are_flagged_and_downgraded`.

---

## Try the diagnosis directly

```bash
curl http://localhost:8000/api/v3/diagnose/COHORT_2026_Q3/OFF2026_01
curl http://localhost:8000/api/v3/trainer/cohort-heatmap/COHORT_2026_Q3
curl http://localhost:8000/api/v3/competencies      # full graph, auditable
```

`data/sample_roster.csv` contains 8 synthetic officers. Uploading it produces a
cohort where ~37% are blocked on the same foundational competency — the
scenario the trainer dashboard is designed to surface.

---

## Scope and honest limitations

**Pilot scope:** one role (Statistical Officer, NSSO), 11 competencies, synthetic data.

| Limitation | Current position |
|---|---|
| No live iGOT integration | REST architecture is integration-ready; no access claimed or implied |
| No real government data | Synthetic data only; production requires authorized access and approval |
| Prerequisite graph is hand-curated | Deliberate — a domain-expert-validated graph is more trustworthy than an inferred one at this scale. Learning it from data is future work |
| Appraisal ingest is structured ratings only | Full appraisal documents are not read or stored |
| Phase 7 is not implemented | Feeding competency data into institutional HR processes requires policy approval, not just code |

---

## Team Disha — Apeejay Stya University

| Member | Role |
|---|---|
| Zaara Khan | Team Lead · ML & Diagnostic Engine |
| Bhupender | Backend & Data Architecture |
| Alok | Frontend & UI/UX |
| Aditi | Competency Taxonomy & Content |
| Dikshita | Research & Evaluation Framework |
| Nikhil | Presentation & Coordination |

---

## License

MIT — see [LICENSE](LICENSE).
