# Live Demo Script (4 minutes)

**Before you start:** backend running on :8000, frontend on :3000, both tabs open.

---

### 1. The setup (30s)
> "This is a cohort of Statistical Officers. Rajesh scored 38% in Data Visualization.
> A normal system would now recommend a visualization course."

### 2. Run the diagnosis (60s)
Click **Trace Root Cause**.

> "The engine walked the prerequisite graph. It found Data Visualization depends on
> Pandas, which depends on Data Handling — and Data Handling is at 42%. So the real
> blocker is two levels upstream. Fixing Data Handling unblocks both downstream skills."

Point at the coverage line.

> "Notice it also tells us we only assessed 5 of 11 competencies — and because some
> prerequisites were never tested, it marks this diagnosis low confidence rather than
> pretending to be certain."

### 3. The conflict case (45s)
Switch to officer OFF2026_02 (Meera).

> "Here the quiz says 45% but the appraisal says 88%. The system doesn't just average
> them into a comfortable 58%. It flags the 43-point divergence, drops confidence, and
> routes it to a trainer — because that disagreement is itself information."

### 4. Cohort view (45s)
Open the **Trainer Dashboard** tab.

> "Across the cohort, 37% are blocked on the same foundational competency. That turns
> eight individual remediation plans into one targeted group workshop."

### 5. Re-assessment (45s)
Click **Simulate Post-Learning Test**.

> "After remediation, we re-test and measure the actual delta. This is the difference
> between 'course completed' and 'competency verified'."

### 6. Close (15s)
> "Every number you just saw came from the live API — nothing on this screen is hardcoded.
> The tests covering these edge cases are in the repo."

---

## If something breaks
- Backend down → show `docs/sample_output.json` and narrate it.
- Always have Swagger (`localhost:8000/docs`) open in a third tab as a fallback.
