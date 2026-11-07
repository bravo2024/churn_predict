# churnpredopt.streamlit.app — CPU / resource-limit diagnosis

**Date:** 2026-11-07
**App:** https://churnpredopt.streamlit.app → repo `bravo2024/churn_predict` (entrypoint `dashboard.py`)
**Local copy:** `N:\Resume Final\company_projects\amazon\amazon_churn_retention`
**Error seen:** "🤯 This app has gone over its resource limits" (Streamlit Community Cloud wording)
**Cloud limits:** CPU 0.078–2 cores, RAM **690 MB minimum – 2.7 GB maximum** (Streamlit docs)

---

## Verdict in one line

It is **not** a broken-package error. The app has one bug that guarantees a retrain on every
cold start, and two footprint problems that make even the "working" path too big:

0. **The artifact download has never worked in production.** `hf_download()` calls
   `hf_hub_download(repo_id="vivekkopthsd/churn-predict-data", ...)` **without
   `repo_type="dataset"`**, so huggingface_hub looks in the *model* namespace, gets
   `401 RepositoryNotFoundError`, the bare `except Exception` swallows it, and
   `load_artifacts()` falls through to `tm.build_artifacts()` — a full in-request retrain of
   RF-200 + GB-200 + XGBoost + Voting + SHAP + RSF-200 + Cox + 5 sensitivity models + a
   20-budget MILP sweep, with `n_jobs=-1`. Verified: the dataset path returns HTTP 200
   (206.7 MB), the model path returns 401.
1. **Even when it loads, 234.4 MB of the 246.6 MB artifact is a Random Survival Forest the
   dashboard never calls.** Loading it pushes the app from 209 MB to **684 MB peak RSS** —
   already at the 690 MB floor before a single chart is drawn.
2. **On first paint the app solves a 24,005-binary MILP built from a 922 MB dense matrix**,
   because the default scope turns survival weighting on. That adds **~1 GB** on top.

Any one of these alone is enough to trigger the page. All three together are deterministic.

The "some packages are breaking" hunch is wrong. Every pin resolves (verified by installing
`requirements.txt` clean: scikit-learn 1.9.1, numpy 2.4.6, scipy 1.16.3, sksurv 0.28.0,
xgboost 3.2.0, shap 0.51.0, streamlit 1.53.1), and the artifact was pickled with exactly
`scikit-learn 1.9.1`. The only real platform/version issue is that **Community Cloud ignores
`.python-version`**.

---

## Evidence (measured, not guessed)

| Measurement | Value | How obtained |
|---|---|---|
| `hf_hub_download(... repo_id="vivekkopthsd/churn-predict-data")` — the app's call | **401 RepositoryNotFoundError** (model namespace) | reproduced on Kaggle + HEAD request |
| Same file via `repo_type="dataset"` | **HTTP 200, 206,660,588 bytes** | HEAD request |
| What the app does after that 401 | `except Exception: return None` → `tm.build_artifacts()` full retrain | `dashboard.py:61-67`, `:112-119` |
| `models/artifacts.pkl` on disk | **246.6 MB** (README says 197 MB) | local file |
| …of which `survival["rsf"]` (RandomSurvivalForest, 200 trees) | **234.4 MB** | per-key `pickle.dumps` |
| …of which everything else combined | **12 MB** | same |
| Does `dashboard.py` ever call the RSF or Cox model? | **No** — it only reads `probs`, the two c-indices, `horizons`, `corr_s90_binary` | grep of `A["survival"]` |
| Is the artifact in the GitHub repo? | **No** — `.gitignore` excludes `models/artifacts.pkl` | repo tree API |
| Cold start: imports only | 209 MB RSS | pinned venv, `psutil` |
| Cold start: imports + full artifact | **684 MB peak / 585 MB resident, 0.6 s** | same |
| Cold start: imports + **slim** artifact (RSF dropped) | **235 MB peak, 0.09 s** | same script, slim pickle |
| Slim artifact on disk | **9.8 MB** | rebuilt and reloaded |
| Artifact pickle `_sklearn_version` | **1.9.1** — matches `scikit-learn==1.9.*` | pickletools scan |
| Test-period customers scored | **4,801** | `models/churn_scores_test.csv` |
| MILP binary variables | **24,005** = 4,801 × 5 offers | arithmetic |
| Dense `A_one` inside `solve_milp_live` | **4,801 × 24,005 float64 = 922 MB** | computed |
| That MILP call: peak RSS / time | **1,077 MB / 6.6 s** | `bench_milp.py` |
| Same MILP with a **sparse** block matrix | **0.192 MB matrix, 157 MB peak, 4.8 s** | `bench_sparse.py` |
| Optimum before vs after sparse rewrite | **identical**: $159,027 retained, ROAS 15.90x, 400 customers | both runs |
| Tabs / charts executed on every rerun | 9 tabs, **25 `st.plotly_chart`**, 8 `st.dataframe` (both from this repo's `dashboard.py`) | grep |

The desktop timings are a floor, not a ceiling: on the cloud the same work runs on ~1 shared
vCPU with Streamlit, plotly, matplotlib, sklearn and xgboost already resident.

---

## Root causes, in order of impact

### 0. The download never worked, so the app retrains on every cold start

`dashboard.py:61-67`:

```python
def hf_download(filename):
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_DATA_REPO, filename=filename)   # no repo_type
    except Exception:
        return None
```

`vivekkopthsd/churn-predict-data` is a **dataset** repo. Without `repo_type="dataset"`,
huggingface_hub resolves it against `https://huggingface.co/api/models/...`, which answers
**401** — reproduced verbatim on Kaggle (`RepositoryNotFoundError: 401 Client Error`, "Invalid
username or password"). The dataset URL answers 200 with 206,660,588 bytes.

The bare `except Exception` hides that, `path` becomes `None`, and `load_artifacts()` falls
into `tm.build_artifacts()`. So on Community Cloud, where `models/artifacts.pkl` does not
exist, **every cold start retrains the whole suite inside the request**: Logistic +
DecisionTree (+ 7-depth search) + RF-200 + GB-200 + XGBoost-200 + Voting + SHAP on 2,000 rows
+ RSF-200 + Cox + five GB(150) sensitivity models + permutation importance, with `n_jobs=-1`,
then a 20-budget MILP sweep. This is multi-minute, 100 % CPU, multi-hundred-MB allocation, and
nothing is persisted — so the next wake-up does it again. That is the "resource limits" page.

Fix: `repo_type="dataset"`, and never convert a download failure into training.

### 1. 234 MB of dead weight: the RSF is shipped but never used

`train_models.build_artifacts()` stores `survival = {"rsf": ..., "cox": ..., "probs": ...}`
with `RandomSurvivalForest(n_estimators=200)`. A fitted sksurv forest pickles its per-node
cumulative-hazard arrays, which is why one model is **234.4 MB**.

`dashboard.py` reads only (`dashboard.py:709-740`, `:837-838`):

```python
sv = A["survival"]
sp = sv["probs"]            # 4,801 x {S_30..S_180, Customer_ID, Expected_Days_To_Return}
sv["cindex_rsf"], sv["cindex_cox"], sv["horizons"], sv["corr_s90_binary"]
```

It never calls `sv["rsf"].predict_*`. All survival output the app needs is already
precomputed into `probs`. Because the file is gitignored, the app downloads those 234 MB
from Hugging Face on **every cold start**, and every 12-hourly wake-up, then unpickles it —
**684 MB peak RSS before rendering**. Dropping `rsf`/`cox` (and the frames the dashboard
does not read) yields a **9.8 MB** artifact that loads in 0.09 s at **235 MB peak**.

### 2. The MILP is written in the worst possible shape and runs on first paint

`dashboard.py:240-266` (`solve_milp_live`):

```python
A_one = np.zeros((n, n * len(offers)))        # 4,801 x 24,005 dense = 922 MB
for i in range(n):
    A_one[i, i*len(offers):(i+1)*len(offers)] = 1.0
res = milp(..., constraints=[LinearConstraint(A_bud, ub=B),
                             LinearConstraint(A_one, ub=1.0)],
           options={"time_limit": 15})
```

The constraint is block diagonal — 5 non-zeros per row — so the dense build is pure waste
and SciPy converts it to sparse anyway. Measured 1,077 MB peak, 6.6 s.

**And it runs before the user touches anything.** Default scope is `index=2` →
`"Churn + Optimization + Survival"` (`dashboard.py:302-306`) → `use_surv=True`
(`:820-824`) → live-solve branch `m = solve_milp_live(ts_off, B)` (`:860`). The most
expensive path in the app executes on the first paint of every cold start.

Commit `ff884b3 "perf: reduce dashboard CPU load for Streamlit Cloud"` lowered the HiGHS
`time_limit` 90 s → 15 s and cached the result. That treats a symptom: the ~1 GB allocation
happens before the solver starts.

### 3. The retraining fallback is a guaranteed kill

```python
def hf_download(filename):
    try: ...
    except Exception:
        return None                      # swallows DNS, 403, timeout, disk-full alike

...
if path:
    return pickle.load(...)
with st.spinner("Artifacts missing - training (~6 min)..."):
    return tm.build_artifacts()          # <-- the killer
```

One download hiccup and the app trains in-request: Logistic + DecisionTree (+ depth search)
+ RF 200 + GB 200 + XGBoost 200 + Voting + SHAP on 2,000 rows + RSF 200 + Cox + five more
GB(150) sensitivity models + permutation importance, with `n_jobs=-1`, then a 20-budget MILP
sweep. On 1–2 shared cores this is multi-minute 100 % CPU with multi-hundred-MB allocations.
Nothing is persisted, so the next cold start repeats it. Both the exception and the training
are invisible behind a spinner — which is exactly why the symptom looks like "CPU overflow".

### 4. All nine tabs execute on every rerun (`st.tabs` is not lazy)

Streamlit reruns the whole script on each interaction; `st.tabs` hides output but does not
skip it. So a budget-slider move re-runs tab 2 (9 figures), tab 3 (PCA/tree/importance),
tab 4 (4 figures), tab 5 (survival), tab 6 (full PCA + KMeans + `silhouette_by_k` = 7 KMeans
fits + 2 figures), tab 7 (MILP + greedy/naive vectorised allocation + figure), tab 8 (power
curves, 2 figures), tab 9 — **25 Plotly figures per interaction**. Throttled CPU plus repeated
allocations is how "slow" becomes "over limits".

### 5. Platform Python is not the Python the repo claims

`.python-version` contains `3.11`, but **Community Cloud does not read it (nor
`runtime.txt`)**; the version is set in *Advanced settings* / app settings, and the platform
default has moved to 3.12/3.13/3.14. Consequences:

* `scikit-survival==0.28.*`, `xgboost==3.2.*`, `shap==0.51.*` (→ `numba`) and `scipy==1.16.*`
  ship wheels per Python minor; on 3.14 you get source builds or import errors.
* If scikit-learn drifts off 1.9.x, `pickle.load` of the 1.9.1 artifact raises, and **there is
  no handler around the unpickle** — that shows as "😦 Oh no", not the resource page.
* Verified pins are real on PyPI and install cleanly together, so the pin set is not the bug.

### 6. Smaller but real

* `import train_models as tm` at the top of `dashboard.py` pulls `xgboost` and `sksurv` into
  every session just to read `FEATURES` — ~180 MB RSS for one list of strings.
* `n_jobs=-1` in RF/RSF oversubscribes a 2-core box.
* Deployed repo is 4 commits behind the local working tree, so a fix must be committed **and**
  pushed or the live app will not change.

---

## Status: implemented and verified (pushed to `main`)

| Change | Where | Verified |
|---|---|---|
| `repo_type="dataset"` in `hf_hub_download`, failure surfaced as a warning | `dashboard.py` `hf_download()` | dataset URL 200 / model URL 401 reproduced |
| Load order slim file → full file → HF, **never trains** | `dashboard.py` `load_artifacts()` | error path stops with a message instead of retraining |
| `models/artifacts_slim.pkl` committed, 9.68 MB, sha256 `8ddea51d…` | repo | 12/12 dashboard keys, 5/5 survival keys, 20 budgets |
| Sparse `A_one` via `kron(speye, …)` | `solve_milp_live()` | 0.192 MB, 4.8 s, optimum identical |
| Survival weighting solved on request (`st.session_state["milp_surv"]`) | tab 7 | click → $84,995 / 8.5x / 400 customers, no exception |
| No `import train_models`; `FEATURES` inlined (24 entries) | top of `dashboard.py` | xgboost/sksurv off the boot path |
| `slim_artifacts()` + dual output | `train_models.py` | writes 206.7 MB and 9.7 MB |
| Test-period scoring switched to the Voting ensemble | `train_models.py` | MILP reproduces the notebook figure |

### Generation note

The figures the CV and README publish come from the `N:\intuit_audit` run and from commit
`ff884b3`: 24 features, 2022-03-19 split, ROC-AUC 0.898, precision@10% 0.487, MILP $202,403 /
20.24x. The uncommitted local working tree held a later rebuild (22 features, 2021 split,
0.901 / 0.473 / $159,027) whose numbers match no published document, so the deployed app was
rebuilt on the published generation instead.

One real defect surfaced while rebuilding: `train_models.py` scored the test period with
**GradientBoosting only**, while the notebook MILP used the **Voting** ensemble. That is why a
faithful rebuild produced $220,409 / 22.04x instead of the published $202,403 / 20.24x. Pointing
the artifact at Voting reproduces the notebook exactly; `train_models.py` on `main` now carries
that scoring, so re-running it regenerates the shipped figures.

Kernel `vivekkopthsd/churn-predict-retrain-generation-b` (version 3) rebuilds everything on
Kaggle from the `ff884b3` tarball and asserts the published figures:

```
Voting  AUC=0.8976  Prec@10%=0.4866          (CV: 0.898 / 0.487)
MILP    rev=$202,403  roas=20.24x  customers=400   (README: ~$202k / ~20x)
sensitivity AUC 0.8665 - 0.8937              (CV: 0.87 - 0.89)
slim artifact: 10.1 MB  sha256 8ddea51db3c9b5e248b86370f6f862f5f98468fc4360c7400ac8306c1b8afe48
contract OK - 12 keys, survival cindex_cox/cindex_rsf/corr_s90_binary/horizons/probs
```

`streamlit.testing.AppTest` on the patched app: **no exception, 9 tabs, 23 charts, 300 MB RSS**,
peak **702 MB**, survival checkbox `False` on first paint.

### Display behaviour

First paint shows the precomputed binary-probability optimum ($202,403 / 20.2x / 400), which is
what the README documents. Survival weighting is one button away and solves on demand.

### Remaining, user-side

1. In the Cloud app: **Reboot app** (the push alone may not cycle the container).
2. Set **Python 3.11** in app settings if it is not already (delete + redeploy; `.python-version`
   is ignored).
3. The Hugging Face copy holds a different build ($220,409 / 22.04x). It is only a fallback now,
   but refresh it so it cannot mislead later.

---

## Plan (original, for reference)

### Phase 0 — the two changes that actually fix it (~1 h)

1. **Ship a slim artifact.** In `train_models.py`, after building the survival dict, persist
   only what the dashboard reads:

   ```python
   slim = {k: v for k, v in artifacts.items()
           if k not in ("tr", "te", "pre", "pre_s", "pca6", "kmeans",
                        "prob_te", "segment_profiles", "seg_names", "silhouette")}
   slim["survival"] = {k: v for k, v in artifacts["survival"].items()
                       if k not in ("rsf", "cox")}      # never called by the app
   ```

   Measured: **246.6 MB → 9.8 MB**, cold start **684 MB → 235 MB peak**, 0.6 s → 0.09 s.
   Keep full artifacts available via a `--full` flag for notebook reproduction.

2. **Commit the 9.8 MB artifact to the repo** and load it from disk first. GitHub's limit is
   100 MB/file, so it fits; the HF download stops being on the critical path at all.
   Keep HF as the fallback for `build_dataset.py` users.

3. **Make the MILP sparse.** Replace the dense block matrix:

   ```python
   from scipy.sparse import csr_matrix, eye as speye, kron

   # one row per customer: sum_d x[i,d] <= 1
   A_one = kron(speye(n, format="csr"), csr_matrix(np.ones((1, len(offers)))))
   ```

   922 MB → 0.19 MB. Verified on the real 4,801-customer table: **same optimum**
   ($159,027 retained, ROAS 15.90x, 400 customers), peak RSS **1,077 → 157 MB**.

4. **Do not solve on first paint.** Default the survival-weighting checkbox to `False` (uses
   the precomputed `OFFER["milp_by_budget"]`) and gate the live solve behind a
   "Compute exact allocation" button stored in `st.session_state`.

5. **Set Python 3.11** in Community Cloud → app settings (delete + redeploy; the file is
   ignored). Add `runtime.txt` with `python-3.11` for other hosts, and a comment in
   `requirements.txt` saying the cloud version is chosen in the UI.

Combined effect on cold start: **684 MB + ~1,000 MB → ~235 MB + ~20 MB**.

### Phase 1 — remove the failure mode (30 min)

6. **Never retrain in a request.** Show a clear message and `st.stop()`; allow
   `build_artifacts()` only when `not ON_CLOUD`, behind an explicit button.
7. Narrow `hf_download`'s `except Exception` to `OSError`/`HTTPError` and log the reason, so a
   download problem is visible in Cloud logs instead of silently becoming a training job.
8. Drop `import train_models` from `dashboard.py`; move `FEATURES` into a 5-line
   `constants.py` so xgboost/sksurv are only imported when training.
9. Set `n_jobs=1` (or 2) in `RandomForestClassifier` / `RandomSurvivalForest` when `ON_CLOUD`.

### Phase 2 — stop recomputing the world (2–3 h)

10. Replace `st.tabs` navigation with `st.radio` / `st.segmented_control` (or `st.fragment`) so
    only the selected section's body runs. 25 figures → ~3–5 per interaction.
11. Cache tab-6 work (PCA, KMeans, `silhouette_by_k`) on `k` and compute the k-sweep once.
12. Trim `requirements.txt` to runtime needs (`papermill`, `nbformat`, `seaborn`,
    `matplotlib` are notebook/build-only; matplotlib is only used for `plot_tree`).

### Phase 3 — keep it fixed (30 min)

13. Log RSS once per cold start (`psutil`) so a regression is visible in Cloud logs.
14. Load test: cold start after **Reboot app**, then 20 slider interactions; RSS flat, no
    "over limits" page.
15. Commit and push (deployed repo is 4 commits behind the local tree).

---

## Acceptance checks

| Check | Target (measured baseline) |
|---|---|
| Artifact on disk / fetched at boot | **9.8 MB committed** (was 246.6 MB downloaded) |
| Cold start peak RSS | **< 300 MB** (was 684 MB) |
| First paint with default scope | no MILP in the request path |
| MILP constraint memory | **0.19 MB sparse** (was 922 MB dense) |
| Peak RSS of a live solve | **157 MB** (was 1,077 MB) |
| Optimum unchanged | $159,027 retained / ROAS 15.90x / 400 customers ✔ |
| Python at runtime | 3.11, as tested |
| 20 slider interactions | flat memory, no re-solve (cache hit) |
| Artifact missing | clear error, no training |

---

## What I would *not* do

* Do not chase package versions. The pins resolve and the artifact matches them exactly; a
  version change here only risks breaking the pickle.
* Do not raise the HiGHS `time_limit` back up, and do not rely on it to bound memory — the
  922 MB allocation happens before the solver starts.
* Do not add `st.cache_data` around more heavy calls before Phase 0; caching a 1 GB
  intermediate keeps the 1 GB resident.
* Do not "compress the pickle" as the primary fix. The file is not dense — it is one unused
  model. Deleting it beats compressing it.
