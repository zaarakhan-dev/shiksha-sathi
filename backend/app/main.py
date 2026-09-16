"""
Shiksha Sathi - Backend API Microservice (v3)
Author: Bhupender (Backend & Database Lead)

v3 changes:
  - Every diagnosis response now carries assessment coverage + confidence.
  - Re-assessment compares the full competency set, not one hard-coded skill.
  - Cohort heatmap reports confidence distribution, not just counts.
  - CORS is explicit rather than wildcard (government deployment expectation).
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import csv
import io
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml")))
from prerequisite_engine import (  # type: ignore[import-not-found]  # noqa: E402
    CompetencyIntelligenceEngine,
    ScoringConfig,
)

app = FastAPI(
    title="Shiksha Sathi Competency Intelligence API",
    description=(
        "Competency diagnosis and prerequisite root-cause tracing for "
        "capacity building (SIH26101). Prototype uses synthetic data."
    ),
    version="3.0.0",
)

ALLOWED_ORIGINS = os.getenv(
    "SS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

engine = CompetencyIntelligenceEngine()


# --------------------------------------------------------------- storage
# In-memory store. Swappable for MongoDB without changing route signatures.

DB: Dict[str, Dict] = {
    "cohorts": {
        "COHORT_2026_Q3": {
            "name": "New Statistical Officers Q3",
            "department": "National Sample Survey Office (NSSO)",
            "officials": {},
        }
    },
    "question_review_queue": [
        {
            "question_id": "Q_LLM_101",
            "competency": "Data_Handling",
            "text": "Which pandas method filters rows using a boolean condition?",
            "options": ["df.loc[]", "df.merge()", "df.groupby()", "df.describe()"],
            "correct_option": "df.loc[]",
            "source_doc": "MoSPI Training Manual Vol 2 - Data Handling.pdf",
            "status": "pending_review",
            "generated_by": "local-llm",
        }
    ],
    "assessment_history": {},
}


def _seed() -> None:
    DB["cohorts"]["COHORT_2026_Q3"]["officials"]["OFF2026_01"] = {
        "id": "OFF2026_01",
        "name": "Rajesh Sharma",
        "role": "Statistical Officer",
        "quiz_scores": {
            "Python_Basics": 80.0,
            "SQL_Basics": 76.0,
            "Data_Handling": 42.0,
            "Pandas_Manipulation": 40.0,
            "Data_Visualization": 38.0,
        },
        "appraisal_scores": {
            "Python_Basics": 80.0,
            "SQL_Basics": 74.0,
            "Data_Handling": 40.0,
            "Pandas_Manipulation": 40.0,
            "Data_Visualization": 40.0,
        },
    }
    # Second officer with a deliberate signal conflict, to demo the flag
    DB["cohorts"]["COHORT_2026_Q3"]["officials"]["OFF2026_02"] = {
        "id": "OFF2026_02",
        "name": "Meera Iyer",
        "role": "Statistical Officer",
        "quiz_scores": {
            "Python_Basics": 45.0,
            "Data_Handling": 48.0,
            "Pandas_Manipulation": 52.0,
        },
        "appraisal_scores": {
            "Python_Basics": 88.0,   # conflict: strong appraisal, weak quiz
            "Data_Handling": 55.0,
        },
    }


_seed()


# --------------------------------------------------------------- models

class ReassessmentRequest(BaseModel):
    cohort_id: str
    official_id: str
    reassessed_quiz_scores: Dict[str, float] = Field(
        ..., description="competency -> new quiz score"
    )


class QuestionReviewRequest(BaseModel):
    question_id: str
    approved: bool
    reviewer: str
    notes: Optional[str] = None


# --------------------------------------------------------------- helpers

def _get_official(cohort_id: str, official_id: str) -> Dict:
    cohort = DB["cohorts"].get(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_id}' not found.")
    official = cohort["officials"].get(official_id)
    if not official:
        raise HTTPException(status_code=404, detail=f"Official '{official_id}' not found.")
    return official


def _diagnose(official: Dict) -> Dict:
    scores = engine.score_all(
        official.get("quiz_scores", {}),
        official.get("appraisal_scores", {}),
    )
    diagnoses = engine.diagnose(scores)
    return {
        "scores": {c: s.to_dict() for c, s in scores.items() if s.is_assessed},
        "diagnoses": [d.to_dict() for d in diagnoses],
        "learning_path": engine.build_learning_path(diagnoses),
        "coverage": engine.coverage_report(scores),
        "conflicts": [s.to_dict() for s in scores.values() if s.conflict],
    }


# --------------------------------------------------------------- routes

@app.get("/")
def root():
    return {
        "service": "Shiksha Sathi Competency Intelligence API",
        "version": "3.0.0",
        "status": "ready",
        "data_notice": "Prototype operating on synthetic data only.",
        "scoring_policy": {
            "quiz_weight": engine.config.quiz_weight,
            "appraisal_weight": engine.config.appraisal_weight,
            "mastery_threshold": engine.config.mastery_threshold,
            "note": "Weights are configurable policy, not fixed constants.",
        },
    }


@app.get("/api/v3/competencies")
def list_competencies():
    """Expose the full competency graph so coverage claims are auditable."""
    return {
        "competencies": engine.all_competencies,
        "prerequisite_edges": [
            {"prerequisite": a, "enables": b} for a, b in engine.dag.edges()
        ],
    }


@app.post("/api/v3/trainer/upload-roster")
async def upload_roster(
    cohort_id: str = Form(...),
    cohort_name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Phase 1 - roster ingest.

    Expected CSV header:
      id,name,role,quiz_<competency>,appraisal_<competency>
    Unrecognised competencies are ignored; missing ones stay unassessed
    rather than being silently defaulted.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv roster files are supported.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Roster must be UTF-8 encoded.")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Roster appears to be empty.")

    known = set(engine.all_competencies)
    DB["cohorts"].setdefault(
        cohort_id,
        {"name": cohort_name, "department": "MoSPI", "officials": {}},
    )

    ingested, skipped = 0, []
    for idx, row in enumerate(reader, start=1):
        off_id = (row.get("id") or "").strip()
        if not off_id:
            skipped.append(f"row {idx}: missing id")
            continue

        quiz, appraisal = {}, {}
        for key, value in row.items():
            if not key or value is None or str(value).strip() == "":
                continue
            if key.startswith("quiz_"):
                comp = key[len("quiz_"):]
                if comp in known:
                    try:
                        quiz[comp] = float(value)
                    except ValueError:
                        skipped.append(f"row {idx}: bad quiz value for {comp}")
            elif key.startswith("appraisal_"):
                comp = key[len("appraisal_"):]
                if comp in known:
                    try:
                        appraisal[comp] = float(value)
                    except ValueError:
                        skipped.append(f"row {idx}: bad appraisal value for {comp}")

        DB["cohorts"][cohort_id]["officials"][off_id] = {
            "id": off_id,
            "name": row.get("name", "Unknown"),
            "role": row.get("role", "Statistical Officer"),
            "quiz_scores": quiz,
            "appraisal_scores": appraisal,
        }
        ingested += 1

    return {
        "status": "success",
        "cohort_id": cohort_id,
        "records_ingested": ingested,
        "warnings": skipped,
    }


@app.get("/api/v3/diagnose/{cohort_id}/{official_id}")
def diagnose_official(cohort_id: str, official_id: str):
    """Phase 3 - composite scoring, root-cause trace, coverage, confidence."""
    official = _get_official(cohort_id, official_id)
    result = _diagnose(official)
    return {
        "official_id": official["id"],
        "name": official["name"],
        "role": official["role"],
        **result,
    }


@app.get("/api/v3/trainer/cohort-heatmap/{cohort_id}")
def cohort_heatmap(cohort_id: str):
    """Phase 6 - surface bottlenecks shared across the cohort."""
    cohort = DB["cohorts"].get(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_id}' not found.")

    officials = cohort["officials"]
    total = len(officials)
    if total == 0:
        return {"cohort_id": cohort_id, "total_officials": 0, "bottlenecks": []}

    counts: Dict[str, int] = {}
    confidence_mix: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    conflict_count = 0

    for off in officials.values():
        res = _diagnose(off)
        conflict_count += len(res["conflicts"])
        seen_roots = set()
        for d in res["diagnoses"]:
            confidence_mix[d["confidence"]] = confidence_mix.get(d["confidence"], 0) + 1
            root = d["root_cause"]
            if root not in seen_roots:      # count each officer once per root
                counts[root] = counts.get(root, 0) + 1
                seen_roots.add(root)

    bottlenecks = [
        {
            "root_competency": root,
            "blocked_officials": n,
            "percentage_of_cohort": round(n / total * 100, 1),
            "insight": f"{round(n / total * 100, 1)}% of the cohort is blocked on '{root}'",
        }
        for root, n in counts.items()
    ]
    bottlenecks.sort(key=lambda b: b["blocked_officials"], reverse=True)

    return {
        "cohort_id": cohort_id,
        "cohort_name": cohort["name"],
        "total_officials": total,
        "bottlenecks": bottlenecks,
        "diagnosis_confidence_mix": confidence_mix,
        "conflicts_requiring_review": conflict_count,
        "suggested_action": (
            f"Consider a group workshop on '{bottlenecks[0]['root_competency']}'."
            if bottlenecks else "No shared bottleneck detected."
        ),
    }


@app.post("/api/v3/reassessment/update")
def reassessment(req: ReassessmentRequest):
    """Phase 5 - measure before/after across every re-tested competency."""
    official = _get_official(req.cohort_id, req.official_id)

    before = dict(official.get("quiz_scores", {}))
    deltas = []
    for comp, new_score in req.reassessed_quiz_scores.items():
        if comp not in engine.all_competencies:
            continue
        old = before.get(comp)
        official.setdefault("quiz_scores", {})[comp] = new_score
        deltas.append({
            "competency": comp,
            "before": old,
            "after": new_score,
            "delta": round(new_score - old, 1) if old is not None else None,
            "newly_assessed": old is None,
        })

    DB["assessment_history"].setdefault(req.official_id, []).append(deltas)
    after = _diagnose(official)

    improved = [d for d in deltas if d["delta"] is not None and d["delta"] > 0]
    return {
        "status": "success",
        "official_id": req.official_id,
        "changes": deltas,
        "competencies_improved": len(improved),
        "remaining_gaps": len(after["diagnoses"]),
        "updated_coverage": after["coverage"],
    }


@app.get("/api/v3/hitl/review-queue")
def review_queue():
    """Phase 4 - AI-generated questions awaiting human approval."""
    pending = [q for q in DB["question_review_queue"] if q["status"] == "pending_review"]
    return {
        "pending_count": len(pending),
        "questions": pending,
        "policy": "No AI-generated item reaches a learner without trainer approval.",
    }


@app.post("/api/v3/hitl/review")
def submit_review(req: QuestionReviewRequest):
    for q in DB["question_review_queue"]:
        if q["question_id"] == req.question_id:
            q["status"] = "approved" if req.approved else "rejected"
            q["reviewed_by"] = req.reviewer
            q["review_notes"] = req.notes
            return {"status": "recorded", "question_id": req.question_id, "new_status": q["status"]}
    raise HTTPException(status_code=404, detail=f"Question '{req.question_id}' not found.")
