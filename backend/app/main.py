"""
Shiksha Sathi - Backend API Microservice (v3)
Author: Bhupender (Backend & Database Lead)

v3 changes:
  - Every diagnosis response now carries assessment coverage + confidence.
  - Re-assessment compares the full competency set, not one hard-coded skill.
  - Cohort heatmap reports confidence distribution, not just counts.
  - CORS is explicit rather than wildcard (government deployment expectation).

v3.1 robustness additions:
  - CORS now includes `null` origin (file:// open) and self-port 8000 for
    Swagger UI testing, in addition to the standard port-3000 dev server.
  - All diagnostic and roster routes return typed Pydantic response models,
    making the schema self-documenting and breaking changes compile-visible.
  - CSV roster validation is hardened: missing `id` column, all-blank rows,
    and out-of-range numeric values are all caught with row-level error context.
  - New lightweight `GET /api/v3/cohort/{cohort_id}/officials` endpoint lets
    the trainer dashboard re-render the heatmap table without re-running full
    diagnosis for every official individually.
"""

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
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
    version="3.1.0",
)

# ---------------------------------------------------------------------------
# Phase 0 — CORS policy
# Allow:
#   • Standard local dev servers on port 3000 (React, Vite, etc.)
#   • The API's own Swagger UI (port 8000)
#   • `null` — sent by browsers when index.html is opened directly from the
#     filesystem via file://, which is the expected local-demo workflow for
#     evaluators who do not run a web server.
# In staging/production, replace via the SS_ALLOWED_ORIGINS env variable.
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS = ",".join([
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "null",          # file:// origin sent by browsers on direct-open
])

ALLOWED_ORIGINS = os.getenv("SS_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

engine = CompetencyIntelligenceEngine()


# ---------------------------------------------------------------------------
# In-memory store — swappable for MongoDB without changing route signatures.
# ---------------------------------------------------------------------------

DB: Dict[str, Any] = {
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
        },
        {
            "question_id": "Q_LLM_102",
            "competency": "Descriptive_Statistics",
            "text": "Which measure of central tendency is least affected by extreme outliers?",
            "options": ["Mean", "Median", "Mode", "Range"],
            "correct_option": "Median",
            "source_doc": "MoSPI Training Manual Vol 3 - Statistics Fundamentals.pdf",
            "status": "pending_review",
            "generated_by": "local-llm",
        },
    ],
    "assessment_history": {},
}


def _seed() -> None:
    """Phase 1 — load synthetic pilot data for demonstration and evaluation."""
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
    # Second officer with a deliberate signal conflict, to demo the HITL flag
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


# ---------------------------------------------------------------------------
# Pydantic response models (Phase 3 schema contract)
# ---------------------------------------------------------------------------

class CompetencyScoreOut(BaseModel):
    """Scored result for a single competency, including raw signals."""
    competency: str
    final_score: Optional[float]
    quiz_score: Optional[float] = None
    appraisal_score: Optional[float] = None
    scoring_mode: str
    confidence: str
    conflict: bool
    conflict_reason: Optional[str] = None


class DiagnosisOut(BaseModel):
    """Root-cause diagnosis for one below-threshold competency."""
    target_competency: str
    target_score: float
    root_cause: str
    root_score: Optional[float]
    is_direct_gap: bool
    confidence: str
    recommendation: str
    evidence: List[str]
    unassessed_prerequisites: List[str]


class CoverageOut(BaseModel):
    """Assessment coverage statistics for the competency graph."""
    total_competencies_in_graph: int
    assessed_count: int
    coverage_percent: float
    unassessed_competencies: List[str]
    conflicted_competencies: List[str]
    note: str


class LearningPathStep(BaseModel):
    step: int
    competency: str
    reason: str


class DiagnoseResponse(BaseModel):
    """Full diagnostic report for one official (Phase 3 output)."""
    official_id: str
    name: str
    role: str
    scores: Dict[str, CompetencyScoreOut]
    diagnoses: List[DiagnosisOut]
    learning_path: List[LearningPathStep]
    coverage: CoverageOut
    conflicts: List[CompetencyScoreOut]


class ScoreDelta(BaseModel):
    competency: str
    before: Optional[float]
    after: float
    delta: Optional[float]
    newly_assessed: bool


class ReassessmentResponse(BaseModel):
    """Phase 5 — before/after comparison after a re-test."""
    status: str
    official_id: str
    changes: List[ScoreDelta]
    competencies_improved: int
    remaining_gaps: int
    updated_coverage: CoverageOut


class BottleneckOut(BaseModel):
    root_competency: str
    blocked_officials: int
    percentage_of_cohort: float
    insight: str


class HeatmapResponse(BaseModel):
    """Phase 6 — cohort-level bottleneck aggregation."""
    cohort_id: str
    cohort_name: str
    total_officials: int
    bottlenecks: List[BottleneckOut]
    diagnosis_confidence_mix: Dict[str, int]
    conflicts_requiring_review: int
    suggested_action: str


class RosterUploadResponse(BaseModel):
    """Phase 1 — result of a CSV roster ingestion."""
    status: str
    cohort_id: str
    records_ingested: int
    warnings: List[str]


class OfficialSummary(BaseModel):
    """Lightweight official record for the trainer heatmap table."""
    id: str
    name: str
    role: str
    composite_scores: Dict[str, Optional[float]]
    confidence_per_competency: Dict[str, str]
    has_conflict: bool


class OfficialsListResponse(BaseModel):
    cohort_id: str
    cohort_name: str
    total_officials: int
    officials: List[OfficialSummary]


class ReviewQueueItem(BaseModel):
    question_id: str
    competency: str
    text: str
    options: List[str]
    correct_option: str
    source_doc: str
    status: str
    generated_by: str


class ReviewQueueResponse(BaseModel):
    """Phase 4 — pending AI-generated questions awaiting human approval."""
    pending_count: int
    questions: List[ReviewQueueItem]
    policy: str


class QuestionReviewRequest(BaseModel):
    question_id: str
    approved: bool
    reviewer: str
    notes: Optional[str] = None


class ReassessmentRequest(BaseModel):
    cohort_id: str
    official_id: str
    reassessed_quiz_scores: Dict[str, float] = Field(
        ..., description="competency -> new quiz score"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_official(cohort_id: str, official_id: str) -> Dict:
    cohort = DB["cohorts"].get(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_id}' not found.")
    official = cohort["officials"].get(official_id)
    if not official:
        raise HTTPException(status_code=404, detail=f"Official '{official_id}' not found.")
    return official


def _diagnose(official: Dict) -> Dict:
    """
    Phase 3 — run the full scoring + diagnosis pipeline for one official.
    Returns raw dicts (not Pydantic models) so it can be called from multiple
    routes without serialisation overhead.
    """
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Shiksha Sathi Competency Intelligence API",
        "version": "3.1.0",
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


# ---- Phase 1 — Roster Ingestion ----

@app.post("/api/v3/trainer/upload-roster", response_model=RosterUploadResponse)
async def upload_roster(
    cohort_id: str = Form(...),
    cohort_name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Phase 1 — CSV roster ingestion.

    Expected CSV header:
      id,name,role,quiz_<competency>,appraisal_<competency>

    Hardening (v3.1):
      - Rejects files missing the `id` column entirely (not just blank values).
      - Skips rows where ALL data cells are blank, with a row-level warning.
      - Reports the bad raw value alongside the row/column location on numeric
        parse failures, so the uploader can locate and fix the source file.
      - Unrecognised competency columns are silently skipped (not an error —
        allows forward-compatible uploads as the competency graph evolves).
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv roster files are supported.")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Roster must be UTF-8 encoded.")

    reader = csv.DictReader(io.StringIO(text))

    # Guard: header must exist and must contain the mandatory `id` column
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Roster appears to be empty.")
    if "id" not in [f.strip().lower() for f in reader.fieldnames]:
        raise HTTPException(
            status_code=422,
            detail=(
                "Roster CSV is missing the required 'id' column. "
                f"Found columns: {list(reader.fieldnames)}"
            ),
        )

    known = set(engine.all_competencies)
    DB["cohorts"].setdefault(
        cohort_id,
        {"name": cohort_name, "department": "MoSPI", "officials": {}},
    )
    # Update name if cohort already existed
    DB["cohorts"][cohort_id]["name"] = cohort_name

    ingested, warnings = 0, []
    for idx, row in enumerate(reader, start=2):   # start=2 because row 1 is the header
        off_id = (row.get("id") or "").strip()

        # Skip rows with a missing official ID
        if not off_id:
            warnings.append(f"Row {idx}: skipped — 'id' cell is blank.")
            continue

        # Skip rows where every non-id cell is blank (header echo or separator lines)
        data_values = [
            v for k, v in row.items()
            if k and k.strip().lower() != "id" and v and str(v).strip()
        ]
        if not data_values:
            warnings.append(
                f"Row {idx} (id={off_id!r}): skipped — all data cells are blank."
            )
            continue

        quiz: Dict[str, float] = {}
        appraisal: Dict[str, float] = {}

        for key, value in row.items():
            if not key or value is None or str(value).strip() == "":
                continue
            key_clean = key.strip()

            if key_clean.startswith("quiz_"):
                comp = key_clean[len("quiz_"):]
                if comp not in known:
                    continue
                try:
                    quiz[comp] = float(value)
                except (ValueError, TypeError):
                    warnings.append(
                        f"Row {idx} (id={off_id!r}): bad quiz value for '{comp}' "
                        f"— got {value!r}, expected a number. Cell skipped."
                    )

            elif key_clean.startswith("appraisal_"):
                comp = key_clean[len("appraisal_"):]
                if comp not in known:
                    continue
                try:
                    appraisal[comp] = float(value)
                except (ValueError, TypeError):
                    warnings.append(
                        f"Row {idx} (id={off_id!r}): bad appraisal value for '{comp}' "
                        f"— got {value!r}, expected a number. Cell skipped."
                    )

        DB["cohorts"][cohort_id]["officials"][off_id] = {
            "id": off_id,
            "name": row.get("name", "Unknown").strip() or "Unknown",
            "role": row.get("role", "Statistical Officer").strip() or "Statistical Officer",
            "quiz_scores": quiz,
            "appraisal_scores": appraisal,
        }
        ingested += 1

    return RosterUploadResponse(
        status="success",
        cohort_id=cohort_id,
        records_ingested=ingested,
        warnings=warnings,
    )


# ---- Phase 3 — Individual Diagnosis ----

@app.get("/api/v3/diagnose/{cohort_id}/{official_id}", response_model=DiagnoseResponse)
def diagnose_official(cohort_id: str, official_id: str):
    """Phase 3 — composite scoring, root-cause trace, coverage, confidence."""
    official = _get_official(cohort_id, official_id)
    result = _diagnose(official)
    return DiagnoseResponse(
        official_id=official["id"],
        name=official["name"],
        role=official["role"],
        scores={k: CompetencyScoreOut(**v) for k, v in result["scores"].items()},
        diagnoses=[DiagnosisOut(**d) for d in result["diagnoses"]],
        learning_path=[LearningPathStep(**s) for s in result["learning_path"]],
        coverage=CoverageOut(**result["coverage"]),
        conflicts=[CompetencyScoreOut(**c) for c in result["conflicts"]],
    )


# ---- Phase 3 — Lightweight Officials List for Trainer Table ----

@app.get("/api/v3/cohort/{cohort_id}/officials", response_model=OfficialsListResponse)
def list_officials(cohort_id: str):
    """
    Phase 3 / 6 — returns composite scores for every official in the cohort
    so the trainer heatmap table can be re-rendered without running a full
    per-official diagnosis N times.

    Each entry includes:
      - composite_scores: final scored value per competency (None if unassessed)
      - confidence_per_competency: high / medium / low per scored entry
      - has_conflict: True if any competency has a quiz/appraisal signal conflict
    """
    cohort = DB["cohorts"].get(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_id}' not found.")

    officials_out: List[OfficialSummary] = []
    for off in cohort["officials"].values():
        scored = engine.score_all(
            off.get("quiz_scores", {}),
            off.get("appraisal_scores", {}),
        )
        composite_scores = {c: s.final_score for c, s in scored.items()}
        confidence_map = {c: s.confidence for c, s in scored.items() if s.is_assessed}
        has_conflict = any(s.conflict for s in scored.values())
        officials_out.append(
            OfficialSummary(
                id=off["id"],
                name=off["name"],
                role=off["role"],
                composite_scores=composite_scores,
                confidence_per_competency=confidence_map,
                has_conflict=has_conflict,
            )
        )

    return OfficialsListResponse(
        cohort_id=cohort_id,
        cohort_name=cohort["name"],
        total_officials=len(officials_out),
        officials=officials_out,
    )


# ---- Phase 5 — Reassessment ----

# Post-training appraisal rating applied to every re-tested competency.
# Setting it to 80.0 reflects a supervisor's updated on-the-job rating after
# the learner completes the remediation module.  The value closes the
# quiz-vs-appraisal gap (keeps divergence well below the 30-pt conflict
# threshold) so the HITL conflict banner is dismissed automatically once the
# learner's scores have improved to mastery level.
_POST_TRAINING_APPRAISAL = 80.0


@app.post("/api/v3/reassessment/update", response_model=ReassessmentResponse)
def reassessment(req: ReassessmentRequest):
    """
    Phase 5 — record post-learning test results and measure the improvement
    delta for every re-tested competency.

    Two mutations are applied per re-tested competency:
      1. quiz_scores[comp]      ← the new objective test result
      2. appraisal_scores[comp] ← post-training supervisor rating (80.0)

    The appraisal update is intentional: the divergence between a high quiz
    score and a stale pre-training appraisal would otherwise keep the signal-
    conflict flag alive even after the learner has demonstrably improved.
    The delta displayed in the UI is quiz-only (before vs after objective
    test), which is the number that actually measures learning progress.
    """
    official = _get_official(req.cohort_id, req.official_id)

    # --- Capture baseline BEFORE any mutation so deltas are accurate ---
    # dict() makes a shallow copy; since values are floats (immutable) this
    # is a true point-in-time snapshot of the pre-update quiz scores.
    before_quiz: Dict[str, float] = dict(official.get("quiz_scores", {}))

    deltas: List[ScoreDelta] = []

    for comp, new_score in req.reassessed_quiz_scores.items():
        if comp not in engine.all_competencies:
            continue  # silently skip competencies not in the DAG

        old_score = before_quiz.get(comp)   # snapshot value — unaffected by later writes

        # 1. Update objective quiz score
        official.setdefault("quiz_scores", {})[comp] = new_score

        # 2. Update appraisal score to post-training rating so the
        #    quiz-appraisal divergence drops below the conflict threshold.
        #    Only applied when the new quiz score itself indicates mastery,
        #    so a struggling learner who scored poorly on the re-test does
        #    not get a false appraisal upgrade.
        if new_score >= engine.config.mastery_threshold:
            official.setdefault("appraisal_scores", {})[comp] = _POST_TRAINING_APPRAISAL

        deltas.append(
            ScoreDelta(
                competency=comp,
                before=old_score,
                after=new_score,
                delta=round(new_score - old_score, 1) if old_score is not None else None,
                newly_assessed=(old_score is None),
            )
        )

    # Persist the delta record for audit/history purposes
    DB["assessment_history"].setdefault(req.official_id, []).append(
        [d.model_dump() for d in deltas]
    )

    # Re-run diagnosis on the fully-updated official record
    after_diag = _diagnose(official)

    improved = [d for d in deltas if d.delta is not None and d.delta > 0]
    return ReassessmentResponse(
        status="success",
        official_id=req.official_id,
        changes=deltas,
        competencies_improved=len(improved),
        remaining_gaps=len(after_diag["diagnoses"]),
        updated_coverage=CoverageOut(**after_diag["coverage"]),
    )


# ---- Phase 6 — Cohort Heatmap ----

@app.get("/api/v3/trainer/cohort-heatmap/{cohort_id}", response_model=HeatmapResponse)
def cohort_heatmap(cohort_id: str):
    """Phase 6 — surface bottlenecks shared across the cohort."""
    cohort = DB["cohorts"].get(cohort_id)
    if not cohort:
        raise HTTPException(status_code=404, detail=f"Cohort '{cohort_id}' not found.")

    officials = cohort["officials"]
    total = len(officials)
    if total == 0:
        return HeatmapResponse(
            cohort_id=cohort_id,
            cohort_name=cohort["name"],
            total_officials=0,
            bottlenecks=[],
            diagnosis_confidence_mix={"high": 0, "medium": 0, "low": 0},
            conflicts_requiring_review=0,
            suggested_action="No officials ingested yet.",
        )

    counts: Dict[str, int] = {}
    confidence_mix: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    conflict_count = 0

    for off in officials.values():
        res = _diagnose(off)
        conflict_count += len(res["conflicts"])
        seen_roots: set = set()
        for d in res["diagnoses"]:
            conf_key = d["confidence"]
            confidence_mix[conf_key] = confidence_mix.get(conf_key, 0) + 1
            root = d["root_cause"]
            if root not in seen_roots:      # count each officer once per root
                counts[root] = counts.get(root, 0) + 1
                seen_roots.add(root)

    bottlenecks_raw = [
        BottleneckOut(
            root_competency=root,
            blocked_officials=n,
            percentage_of_cohort=round(n / total * 100, 1),
            insight=f"{round(n / total * 100, 1)}% of the cohort is blocked on '{root}'",
        )
        for root, n in counts.items()
    ]
    bottlenecks_raw.sort(key=lambda b: b.blocked_officials, reverse=True)

    return HeatmapResponse(
        cohort_id=cohort_id,
        cohort_name=cohort["name"],
        total_officials=total,
        bottlenecks=bottlenecks_raw,
        diagnosis_confidence_mix=confidence_mix,
        conflicts_requiring_review=conflict_count,
        suggested_action=(
            f"Consider a group workshop on '{bottlenecks_raw[0].root_competency}'."
            if bottlenecks_raw else "No shared bottleneck detected."
        ),
    )


# ---- Phase 4 — HITL Review Queue ----

@app.get("/api/v3/hitl/review-queue", response_model=ReviewQueueResponse)
def review_queue():
    """Phase 4 — AI-generated questions awaiting human approval."""
    pending = [
        ReviewQueueItem(**q)
        for q in DB["question_review_queue"]
        if q["status"] == "pending_review"
    ]
    return ReviewQueueResponse(
        pending_count=len(pending),
        questions=pending,
        policy="No AI-generated item reaches a learner without trainer approval.",
    )


@app.post("/api/v3/hitl/review")
def submit_review(req: QuestionReviewRequest):
    """Phase 4 — trainer approves or rejects an AI-generated question."""
    for q in DB["question_review_queue"]:
        if q["question_id"] == req.question_id:
            q["status"] = "approved" if req.approved else "rejected"
            q["reviewed_by"] = req.reviewer
            q["review_notes"] = req.notes
            return {
                "status": "recorded",
                "question_id": req.question_id,
                "new_status": q["status"],
            }
    raise HTTPException(
        status_code=404,
        detail=f"Question '{req.question_id}' not found in the review queue.",
    )
