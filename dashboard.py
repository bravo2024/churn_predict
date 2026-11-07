"""
Customer Retention Analytics - Business Analytics Dashboard.

Tabs follow the analysis notebook step by step. A scope selector in the sidebar
chooses which modules appear - Churn, Churn + Survival, or Churn + Optimization +
Survival. Core logic (temporal validation, models, MILP, survival) is unchanged from
the validated pipeline. Text is brief, third person, passive voice, no colons or
semicolons, white background with black text.

Run:  streamlit run dashboard.py
"""
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import stats
from scipy.sparse import csr_matrix, eye as speye, kron

import os as _os
ON_CLOUD = (_os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"
            or "streamlit.app" in _os.environ.get("HOSTNAME", ""))

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "models", "artifacts.pkl")
SLIM_ARTIFACTS = os.path.join(BASE, "models", "artifacts_slim.pkl")
TRAIN_CSV = os.path.join(BASE, "data", "snapshot_train.csv")
TEST_CSV = os.path.join(BASE, "data", "snapshot_test.csv")
# Kept in sync with train_models.FEATURES. Duplicated here so importing xgboost and
# scikit-survival (about 180 MB of RSS) is not on the app's cold-start path.
FEATURES = ["orders_12m", "spend_12m", "aov_12m", "aov_recent", "aov_prior", "aov_trend",
            "recency_days", "active_months_12m", "category_count_12m", "avg_price_12m",
            "entertainment_share", "gaming_share", "media_share", "gift_card_share", "ott_flag",
            "Age", "Annual_Income_USD", "Household_Size", "Reported_Usage_Freq",
            "Gender", "Education_Level", "Region"]
# Artifacts are too large for GitHub (197 MB) and are hosted on Hugging Face.
HF_DATA_REPO = "vivekkopthsd/churn-predict-data"


def hf_download(filename):
    """Download `filename` from the companion HF dataset; returns a local path or None.

    repo_type="dataset" is required. Without it huggingface_hub looks for a *model* repo of
    the same name, receives 401, and the caller used to fall through to a full retrain - which
    is what pushed the Community Cloud container over its resource limits.
    """
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_DATA_REPO, filename=filename, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 - surfaced, never converted into training
        st.warning(f"Hugging Face download failed for {filename}: {type(exc).__name__}")
        return None

# ---------------------------------------------------------------- design
NAVY, BLUE, GREEN, AMBER, RED, GRAY = "#1f3a5f", "#2b5c8f", "#2e7d32", "#b26a00", "#c62828", "#64748b"
st.set_page_config(page_title="Customer Retention Analytics", page_icon="📊", layout="wide")
st.markdown(f"""
<style>
.stApp{{background:#ffffff;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:1.05rem;color:#111;}}
.block-container{{padding-top:1.2rem;max-width:1440px;}}
h1,h2,h3{{color:#000;}}
.tag{{font-size:.8rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#000;
 border-left:4px solid {BLUE};padding-left:8px;margin:20px 0 8px 0;}}
.metric-card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 12px;
 box-shadow:0 1px 3px rgba(15,23,42,.06);text-align:center;height:100%;}}
.metric-value{{font-size:1.85rem;font-weight:700;color:#000;line-height:1.15;}}
.metric-label{{font-size:.8rem;color:{GRAY};text-transform:uppercase;letter-spacing:.05em;margin-top:3px;}}
.metric-sub{{font-size:.8rem;font-weight:600;margin-top:5px;}}
.doing{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;
 margin:6px 0 12px 0;font-size:.98rem;}}
.interpret{{background:#f4f4f4;border-left:3px solid {BLUE};border-radius:0 8px 8px 0;
 padding:8px 14px;margin:8px 0;font-size:.95rem;color:#111;}}
div[data-testid="stTabs"] button[data-baseweb="tab"]{{font-size:.82rem;padding:6px 12px;}}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{{gap:4px;}}
.amz-card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;margin:8px 0;
 box-shadow:0 1px 2px rgba(0,0,0,.05);}}
.amz-card-title{{font-size:.95rem;font-weight:700;color:#0f1111;margin-bottom:4px;}}
.amz-card-text{{font-size:.86rem;color:#111;line-height:1.4;}}
.amz-footer{{font-size:.72rem;color:#767676;margin-top:12px;padding-top:8px;border-top:1px solid #e7e7e7;}}
</style>
""", unsafe_allow_html=True)


def doing(text):
    st.markdown(f'<div class="doing"><b>What is being done</b><br/>{text}</div>', unsafe_allow_html=True)


def tag(text):
    st.markdown(f'<div class="tag">{text}</div>', unsafe_allow_html=True)


def metric_card(col, value, label, sub=None, sub_color=GREEN):
    sub_html = f'<div class="metric-sub" style="color:{sub_color}">{sub}</div>' if sub else ""
    col.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div>'
                 f'<div class="metric-label">{label}</div>{sub_html}</div>', unsafe_allow_html=True)


def interp(text):
    st.markdown(f'<div class="interpret">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- core logic (preserved)
@st.cache_resource(show_spinner="Loading pre-trained suite...")
def load_artifacts():
    """Load the precomputed suite, largest-first avoidance.

    Order: committed slim file, full local file, slim from HF, full from HF.
    The app never trains inside a request - on Community Cloud that retraining is what
    pushes the container over its resource limits. Rebuild with `python train_models.py`.
    """
    for path in (SLIM_ARTIFACTS, ARTIFACTS):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    for name in ("models/artifacts_slim.pkl", "models/artifacts.pkl"):
        path = hf_download(name)
        if path:
            with open(path, "rb") as f:
                return pickle.load(f)
    st.error("Precomputed artifacts are unavailable, so the dashboard cannot start. "
             "Training inside a request exceeds the Community Cloud resource limits, so it "
             "is disabled here. Rebuild with `python train_models.py` and commit "
             "models/artifacts_slim.pkl.")
    st.stop()


@st.cache_data(show_spinner="Loading temporal snapshots...")
def load_snapshots():
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def make_pre(df):
    num = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
    cat = df[FEATURES].select_dtypes(include=["object"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))]), cat)])
    return pre.fit_transform(df[FEATURES]), pre


def prec_at_top_decile(y, prob):
    k = max(1, len(y) // 10)
    return float(y.iloc[np.argsort(-prob)[:k]].mean())


@st.cache_data(show_spinner=False)
def silhouette_by_k(Xp6):
    return {k: float(silhouette_score(Xp6, KMeans(k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xp6)))
            for k in range(2, 9)}


@st.cache_resource(show_spinner="Training the selected model (temporal validation)...")
def train_churn(algo, window, max_depth, n_estimators, lr, C):
    """Train a real model on the temporal train period; evaluate on the unseen test period."""
    tr, te = load_snapshots()
    yl = f"label_churn_{window}"
    y_tr, y_te = tr[yl], te[yl]
    Xp_tr, pre = make_pre(tr)
    Xp_te = pre.transform(te[FEATURES])

    t0 = time.time()
    if algo == "Logistic Regression":
        m = LogisticRegression(max_iter=2000, C=C, random_state=RANDOM_STATE).fit(Xp_tr, y_tr)
    elif algo == "Decision Tree":
        m = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(Xp_tr, y_tr)
    elif algo == "Random Forest":
        m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE, n_jobs=-1).fit(Xp_tr, y_tr)
    elif algo == "Gradient Boosting":
        m = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=RANDOM_STATE).fit(Xp_tr, y_tr)
    else:  # Voting (Logistic + RF + GB)
        base = [LogisticRegression(max_iter=2000, C=C, random_state=RANDOM_STATE),
                RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE, n_jobs=-1),
                GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=RANDOM_STATE)]
        m = VotingClassifier([("lr", base[0]), ("rf", base[1]), ("gb", base[2])], voting="soft").fit(Xp_tr, y_tr)
    train_s = time.time() - t0

    prob = m.predict_proba(Xp_te)[:, 1]
    p = (prob >= 0.5).astype(int)
    metrics = {
        "ROC-AUC": round(float(roc_auc_score(y_te, prob)), 4),
        "Prec@10%": round(prec_at_top_decile(y_te, prob), 4),
        "Accuracy": round(float(accuracy_score(y_te, p)), 4),
        "Precision": round(float(precision_score(y_te, p)), 4),
        "Recall": round(float(recall_score(y_te, p)), 4),
        "F1": round(float(f1_score(y_te, p)), 4),
    }
    names = FEATURES if algo == "Logistic Regression" else None
    imp = None
    if hasattr(m, "feature_importances_"):
        num = [f for f in FEATURES if tr[f].dtype.kind in "iuf"]
        cat = [f for f in FEATURES if tr[f].dtype.kind == "O"]
        names = num + list(pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat))
        imp = pd.Series(m.feature_importances_, index=names).sort_values(ascending=False).head(12)
    elif algo == "Logistic Regression":
        num = [f for f in FEATURES if tr[f].dtype.kind in "iuf"]
        cat = [f for f in FEATURES if tr[f].dtype.kind == "O"]
        names = num + list(pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat))
        imp = pd.Series(np.abs(m.coef_[0]), index=names).sort_values(ascending=False).head(12)
    return {"metrics": metrics, "imp": imp, "train_s": train_s, "prob": prob, "y_te": y_te,
            "base_rate": float(y_te.mean()), "conf": confusion_matrix(y_te, p), "algo": algo,
            "model": m, "names": names}


@st.cache_data(show_spinner=False)
def voting_test_confusion():
    tr, te = load_snapshots()
    Xp_tr, pre = make_pre(tr)
    Xp_te = pre.transform(te[FEATURES])
    y = te["label_churn"].values
    p = A["fitted_models"]["Voting (Ch7)"].predict(Xp_te)
    return confusion_matrix(y, p)


@st.cache_data(show_spinner=False)
def train_test_auc():
    tr, te = load_snapshots()
    Xp_tr, pre = make_pre(tr)
    Xp_te = pre.transform(te[FEATURES])
    y_tr, y_te = tr["label_churn"], te["label_churn"]
    rows = []
    for name in ["Logistic (Ch4)", "DecisionTree (Ch6)", "RandomForest (Ch7)",
                 "GradientBoosting (Ch7)", "XGBoost (Ch7)", "Voting (Ch7)"]:
        m = A["fitted_models"][name]
        rows.append({"Model": name.replace(" (Ch4)", "").replace(" (Ch6)", "").replace(" (Ch7)", ""),
                     "Train AUC": round(float(roc_auc_score(y_tr, m.predict_proba(Xp_tr)[:, 1])), 3),
                     "Test AUC": round(float(roc_auc_score(y_te, m.predict_proba(Xp_te)[:, 1])), 3)})
    df = pd.DataFrame(rows)
    df["Gap"] = (df["Train AUC"] - df["Test AUC"]).round(3)
    return df.sort_values("Test AUC", ascending=False)


@st.cache_data(show_spinner=False)
def per_model_importance():
    """Top features by importance or coefficient magnitude, per fitted model."""
    out = {}
    for name, m in A["fitted_models"].items():
        if hasattr(m, "feature_importances_"):
            out[name] = pd.Series(m.feature_importances_, index=A["feature_names"]).sort_values(ascending=False).head(10)
        elif hasattr(m, "coef_"):
            out[name] = pd.Series(np.abs(m.coef_[0]), index=A["feature_names"]).sort_values(ascending=False).head(10)
    return out


@st.cache_data(show_spinner="Solving MILP...")
def solve_milp_live(ts, B, offers=(5, 10, 15, 20, 25), base=0.04, slope=0.018, cap=0.60):
    """Exact 0/1-knapsack allocation for the current scoring mode (survival or binary)."""
    def r(d):
        return float(np.clip(base + slope * d, 0.0, cap))
    r0 = r(0)
    p = ts["Churn_Prob"].values
    V = ts["CLV_3Yr"].values
    U = ts["urgency"].values
    DV = np.array([[pi * Vi * U[i] * (r(d) - r0) for d in offers] for i, (pi, Vi) in enumerate(zip(p, V))])
    n = len(ts)
    A_bud = np.tile(np.array(offers, dtype=float), n)
    # Each customer is one row with five non-zeros. Stored sparse: 0.19 MB, not 922 MB.
    A_one = kron(speye(n, format="csr"), csr_matrix(np.ones((1, len(offers)))))
    res = milp(c=-DV.ravel(), integrality=np.ones(n * len(offers)), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(A_bud, ub=B), LinearConstraint(A_one, ub=1.0)],
               options={"time_limit": 15})
    x = np.round(res.x).astype(int).reshape(n, len(offers))
    al = ts[["Customer_ID", "Churn_Prob", "CLV_3Yr"]].copy()
    al["Offer_USD"] = (x * np.array(offers)).sum(axis=1)
    al["Incr_Retained_Rev"] = (x * DV).sum(axis=1)
    al = al[al.Offer_USD > 0].sort_values("Incr_Retained_Rev", ascending=False).reset_index(drop=True)
    return {"allocation": al, "rev": float(al.Incr_Retained_Rev.sum()),
            "roas": float(al.Incr_Retained_Rev.sum() / max(al.Offer_USD.sum(), 1)),
            "customers": int(len(al))}


A = load_artifacts()
tr, te = load_snapshots()
y_tr = tr["label_churn"]
y_te = te["label_churn"]
cust = A["cust"]
ts = A["test_scores"]
OFFER = A["offer_result"]
BASE_RATE = float(te.label_churn.mean())
_top = ts.sort_values("Churn_Prob", ascending=False).head(max(1, len(ts) // 10))
REV_AT_RISK, N_AT_RISK = float(_top["CLV_3Yr"].sum()), int(len(_top))

# ---------------------------------------------------------------- header
st.title("Customer Churn Prediction and Retention-Budget Optimization")
st.caption("Customer Retention Analytics  |  Real e-commerce data  |  Temporal validation  |  Exact MILP budget allocation")
st.caption("This dashboard follows the analysis notebook step by step. Use the analysis scope "
           "in the sidebar to choose the modules - churn only, churn plus survival, or churn "
           "plus optimization plus survival.")

tag("BUSINESS PROBLEM")
st.markdown("Retention is the highest-leverage growth lever in e-commerce. The objective is to "
            "predict which customers will stop purchasing within the next 90 days, explain why "
            "they disengage, and allocate a limited retention budget so that every dollar "
            "protects the most customer value.")

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### Customer Retention Analytics")
st.sidebar.caption("Churn | Survival | Budget optimization")
scope = st.sidebar.radio("Analysis scope",
                         ["Churn", "Churn + Survival", "Churn + Optimization + Survival"],
                         index=2,
                         help="Churn shows the binary model. Survival adds time-to-return. "
                              "Optimization adds the budget layer with survival weighting.")
st.sidebar.markdown(f'<div class="amz-card">'
                    f'<div class="amz-card-title">Key result</div>'
                    f'<div class="amz-card-text">4.95x precision at top 10%. MILP returns 15.9x per '
                    f'dollar.</div></div>', unsafe_allow_html=True)

with st.sidebar.expander("Data"):
    st.markdown("- Open E-Commerce 1.0 (MIT), 1.76M purchases + 5,027 survey -> 5,013 customers")
    st.markdown("- Train features <= 2021-03-19, test on unseen 2021-09-19")
    st.markdown("- No leakage. Pre-cutoff features. CLV winsorized at p99")
with st.sidebar.expander("Method"):
    st.markdown("- Churn = no purchase in N days (90 used, AUC stable 0.87-0.90)")
    st.markdown("- Models: Logistic, Trees, RF, GB, XGBoost, Voting")
    st.markdown("- Survival: RSF and Cox model time to return")
    st.markdown("- Offers: exact 0/1-knapsack MILP (HiGHS)")
with st.sidebar.expander("Scope"):
    st.markdown("- Panel: US heavy users, median 221 orders")
    st.markdown("- CLV = annualized spend x 3 (ranking measure)")
    st.markdown("- Response curve assumed, validated by A/B test")
    st.markdown("- Single cross-section, rolling retrain recommended")
st.sidebar.markdown('<div class="amz-footer">Back to top  |  Retention Analytics  |  2026</div>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- tabs (scope-driven)
tab_names = ["1 · Problem & Data", "2 · Exploratory", "3 · Churn Model", "4 · Importance"]
if "Survival" in scope:
    tab_names += ["5 · Survival"]
tab_names += ["6 · Segments", "7 · Optimizer (MILP)", "8 · A/B Test", "9 · Recommendations"]
T = st.tabs(tab_names)
IDX = {n: i for i, n in enumerate(tab_names)}

# ================================================================ 1
with T[IDX["1 · Problem & Data"]]:
    doing("The business question is defined, success criteria are set, and a temporally "
          "validated dataset is prepared. The full problem statement and dataset description "
          "are shown above the tabs.")
    st.markdown("**Success criteria.** ROC-AUC at least 0.85 on the unseen test period, "
                "Precision at top-10% at least 3x the base churn rate, stable AUC across 30-180 "
                "day windows, and an allocation that beats naive spend.")

    tag("DATASETS USED")
    c1, c2, c3 = st.columns(3)
    metric_card(c1, "1.76M purchases", "Purchases 2018-2023", "price, category, state, asin, survey link", NAVY)
    metric_card(c2, "5,027 respondents", "Demographics survey", "age, income, gender, education, region", BLUE)
    metric_card(c3, "5,013 customers", "Derived customer table", "behaviour and demographics, 27 columns", GRAY)
    st.markdown("The analysis uses the public Open E-Commerce 1.0 release (Berke et al., 2024, "
                "MIT Media Lab) - 1.76 million real Amazon purchases from 2018 to 2023, linked to "
                "a 5,027-respondent demographics survey, joined at 99.96 percent and aggregated "
                "into a 5,013-customer analysis table.")

    tag("CUSTOMER PROFILE")
    st.markdown("The panel consists of US Amazon shoppers recruited through an online survey. "
                "They are frequent buyers, with a median of 221 orders per customer over the "
                "observation window. The 90-day churn base rate is 9.6 percent within this "
                "engaged, high-activity cohort, and the results describe this panel rather than "
                "occasional shoppers.")

    tag("DATA SPLIT (TEMPORAL)")
    c1, c2, c3 = st.columns(3)
    metric_card(c1, f"{len(tr):,}", "Train customers", "features up to 2021-03-19", NAVY)
    metric_card(c2, f"{len(te):,}", "Test customers", "unseen period from 2021-09-19", BLUE)
    metric_card(c3, f"{BASE_RATE:.1%}", "Churn base rate 90d", "test period", AMBER)
    interp("Train and test customers are disjoint in time. The model is built on the train "
           "period only and measured on the test period, which it never saw.")

    tag("TEMPORAL VALIDATION")
    c1, c2 = st.columns(2)
    c1.markdown("**Train period** - features up to **2021-03-19**. The label is no purchase in "
                "the next 90 days, fully observed to 2021-06-17.")
    c2.markdown("**Test period** - features up to **2021-09-19** with the same 90-day label. "
                "The two periods are disjoint in time. The model never sees the future and no "
                "outcome window is censored.")
    interp("This design is why the reported metrics are trustworthy. They measure prediction "
           "of the future, not reconstruction of the past.")
    interp("Why these cutoffs: each respondent's purchase history ends at their survey "
           "submission, so a later cutoff (e.g. 2022-09-19) mistakes survey-submission "
           "truncation for churn - 76.5 percent of labelled churners had no data after that "
           "cutoff at all, versus 14.2 percent at 2021-09-19, which is consistent with genuine "
           "permanent lapse.")

# ================================================================ 2
with T[IDX["2 · Exploratory"]]:
    doing("Customer behaviour and the value at stake are explored before any model is built.")

    c1, c2 = st.columns(2)
    with c1:
        tag("CHURN RATE BY DEFINITION WINDOW")
        fig = px.line(x=["30d", "60d", "90d", "120d", "180d"],
                      y=[tr[f"label_churn_{w}"].mean() for w in (30, 60, 90, 120, 180)],
                      markers=True, template="plotly_white", labels={"x": "Window", "y": "Churn rate"})
        fig.update_traces(line_color=BLUE, marker_color=BLUE)
        st.plotly_chart(fig, width='stretch')
        interp("90 days balances catching real churn against ignoring normal gaps.")
    with c2:
        tag("VALUE AT STAKE (CLV)")
        fig = px.histogram(cust, x="CLV_3Yr", nbins=40, template="plotly_white",
                           labels={"CLV_3Yr": "3-year CLV ($)"}, color_discrete_sequence=[BLUE])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        interp(f"The pool is ${cust['CLV_3Yr'].sum():,.0f} and the top 10% of customers hold 34% "
               f"of it. Value is concentrated, so retention spend should be too.")

    c1, c2 = st.columns(2)
    with c1:
        tag("CLASS BALANCE")
        bal = pd.DataFrame({
            "Period": ["Train", "Train", "Test", "Test"],
            "Class": ["Returned", "Churned", "Returned", "Churned"],
            "Customers": [int(tr.label_churn.eq(0).sum()), int(tr.label_churn.sum()),
                          int(te.label_churn.eq(0).sum()), int(te.label_churn.sum())]})
        fig = px.bar(bal, x="Period", y="Customers", color="Class", barmode="group",
                     template="plotly_white",
                     color_discrete_map={"Returned": BLUE, "Churned": RED}, text_auto=".2s")
        st.plotly_chart(fig, width='stretch')
        interp(f"Churn is rare - {tr.label_churn.mean():.1%} in train and "
               f"{te.label_churn.mean():.1%} in test. This is why Precision at top-10% is the "
               f"business metric rather than accuracy.")
    with c2:
        tag("OTT / STREAMING BUYERS")
        ott_df = te.groupby("ott_flag")["label_churn"].mean().rename(
            index={0: "No OTT / streaming", 1: "Bought OTT / streaming"}).rename_axis("group").reset_index()
        fig = px.bar(ott_df, x="group", y="label_churn", template="plotly_white",
                     labels={"group": "", "label_churn": "90d churn rate"}, text_auto=".1%",
                     color_discrete_sequence=[BLUE])
        st.plotly_chart(fig, width='stretch')
        interp("Subscription-style buyers churn about 3x less. They are a cross-sell audience.")

    c1, c2 = st.columns(2)
    with c1:
        tag("AVERAGE ORDER VALUE TREND")
        fig = px.box(te, x="label_churn", y="aov_trend", template="plotly_white",
                     labels={"label_churn": "90-day outcome", "aov_trend": "AOV trend ($)"},
                     color="label_churn", color_discrete_map={0: BLUE, 1: RED})
        st.plotly_chart(fig, width='stretch')
        interp("A falling basket size precedes churn. It is a CRM early-warning signal.")
    with c2:
        tag("CORRELATES OF CHURN")
        cc = tr[["label_churn", "recency_days", "orders_12m", "spend_12m", "active_months_12m",
                 "category_count_12m", "aov_trend", "ott_flag", "Annual_Income_USD", "Age"]].corr()["label_churn"].drop("label_churn").sort_values()
        fig = px.bar(x=cc.values, y=cc.index, orientation="h", template="plotly_white",
                     labels={"x": "Correlation with 90d churn", "y": ""},
                     color=np.sign(cc.values),
                     color_discrete_map={-1.0: GREEN, 1.0: RED}).update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        interp("Rising recency raises churn risk. Activity, spend and category breadth protect against it.")

    tag("CORRELATES OF CLV")
    cc2 = cust[["CLV_3Yr", "Total_Spend_USD", "Order_Count", "Avg_Order_Value_USD", "Orders_Per_Year",
                "Active_Months", "Category_Count", "Recency_Days", "Annual_Income_USD", "entertainment_share"]].corr()["CLV_3Yr"].drop("CLV_3Yr").sort_values()
    fig = px.bar(x=cc2.values, y=cc2.index, orientation="h", template="plotly_white",
                 labels={"x": "Correlation with CLV", "y": ""},
                 color=np.sign(cc2.values),
                 color_discrete_map={-1.0: GREEN, 1.0: RED}).update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')
    interp("Value follows spend, order count and category breadth, not demographics.")

    tag("RFM SCORE AND QUADRANT")
    rfm = cust[["Customer_ID", "Recency_Days", "Order_Count", "Total_Spend_USD"]].copy()
    z = lambda s: (s - s.mean()) / s.std()
    rfm["R"] = z(rfm["Recency_Days"])
    rfm["F"] = z(np.log1p(rfm["Order_Count"]))
    rfm["M"] = z(np.log1p(rfm["Total_Spend_USD"]))
    rfm["RFM_Score"] = (-rfm["R"] + rfm["F"] + rfm["M"]).round(2)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(rfm, x="RFM_Score", nbins=40, template="plotly_white",
                           color_discrete_sequence=[BLUE]).update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        interp("The composite combines recency, frequency and monetary value. Higher score "
               "means a more engaged customer.")
    with c2:
        fig = px.scatter(rfm, x="Recency_Days", y="Order_Count", color="Total_Spend_USD",
                         log_y=True, template="plotly_white",
                         labels={"Recency_Days": "Recency (days)", "Order_Count": "Orders",
                                 "Total_Spend_USD": "Total spend"})
        st.plotly_chart(fig, width='stretch')
        interp("The quadrant shows value concentration. Recency, frequency and monetary are "
               "the churn-model predictors. The composite score is used for analysis and "
               "segmentation only, not added to the model, because it is perfectly collinear "
               "with its three components.")

# ================================================================ 3
with T[IDX["3 · Churn Model"]]:
    doing("A custom model is trained on the train period and evaluated on the unseen test "
          "period. The algorithm and hyperparameters are chosen here, and the full metrics "
          "are shown.")
    st.caption(f"Features per customer: {len(FEATURES)}. Train and test sizes and the base "
               f"rate are shown in tab 1.")

    with st.form("model_lab_form"):
        c1, c2, c3 = st.columns(3)
        algo = c1.selectbox("Algorithm", ["Voting (LR + RF + GB)", "Gradient Boosting", "Random Forest",
                                          "Decision Tree", "Logistic Regression", "XGBoost"])
        window = c2.selectbox("Churn window (days)", [30, 60, 90, 120, 180], index=2)
        C = c3.selectbox("Regularization C (logistic)", [0.01, 0.1, 1.0, 10.0], index=2)
        c4, c5, c6 = st.columns(3)
        max_depth = c4.slider("Max tree depth", 2, 10, 4)
        n_estimators = c5.slider("Number of trees / estimators", 50, 300, 200, 50)
        lr = c6.slider("Learning rate (boosting)", 0.01, 0.20, 0.08, 0.01)
        submitted = st.form_submit_button("Train and Evaluate", type="primary")

    if submitted:
        st.session_state["lab"] = train_churn(algo, window, max_depth, n_estimators, lr, C)
        st.session_state["lab_cfg"] = {"algo": algo, "window": window, "depth": max_depth,
                                       "trees": n_estimators, "lr": lr, "C": C}

    lab = st.session_state.get("lab")
    if lab is not None:
        cfg = st.session_state["lab_cfg"]
        st.success(f"Trained {cfg['algo']} on a {cfg['window']}-day window in {lab['train_s']:.1f}s.")
        m = lab["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, f'{m["ROC-AUC"]:.4f}', "ROC-AUC", "0.5 random, 0.9 excellent", BLUE)
        metric_card(c2, f'{m["Prec@10%"]:.1%}', "Precision at top 10%", f"base rate {lab['base_rate']:.1%}", GREEN)
        metric_card(c3, f'{m["Precision"]:.3f}', "Precision", f"recall {m['Recall']:.3f}", GRAY)
        metric_card(c4, f'{m["F1"]:.3f}', "F1", f"accuracy {m['Accuracy']:.3f}", GRAY)
        c1, c2 = st.columns(2)
        c1.markdown("**Confusion matrix (test period)**")
        c1.plotly_chart(px.imshow(lab["conf"], text_auto=True, color_continuous_scale="Blues",
                                  template="plotly_white", labels=dict(x="Predicted", y="Actual"),
                                  x=["Returned", "Churned"], y=["Returned", "Churned"]), width='stretch')
        if lab["imp"] is not None:
            c2.markdown("**Feature importance (this model)**")
            fig = px.bar(lab["imp"].head(10).reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
                         x="Importance", y="Feature", orientation="h", template="plotly_white",
                         color="Importance", color_continuous_scale="Blues").update_layout(yaxis={"autorange": "reversed"})
            c2.plotly_chart(fig, width='stretch')
        interp(f"ROC-AUC {m['ROC-AUC']:.3f} with Precision at top-10% of {m['Prec@10%']:.0%} means "
               f"the riskiest decile is contacted first. That is the efficient way to spend a budget.")
        if cfg["algo"] == "Decision Tree" and lab.get("model") is not None:
            st.markdown("**Decision tree structure (this model)**")
            fig_t, ax_t = plt.subplots(figsize=(13, 6))
            plot_tree(lab["model"], feature_names=lab.get("names"), max_depth=3, filled=True,
                      fontsize=8, ax=ax_t)
            st.pyplot(fig_t)
            interp("A shallow tree is a readable rule book. Each split shows the feature, the "
                   "threshold, and the churn probability in the node.")
    else:
        st.info("No custom model has been trained yet. The pre-trained benchmark is shown below, "
                "or set the controls and click Train and Evaluate.")
        cs = A["churn_summary"]
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, f"{cs['test_customers']:,}", "Customers in test period", "unseen 2021-09-19 window", NAVY)
        metric_card(c2, f"{cs['actual_churned']:,}", "Actually churned", f"base rate {cs['base_rate']:.1%}", RED)
        metric_card(c3, f"{cs['predicted_churned_05']:,}", "Predicted churned (0.5 cutoff)", "XGBoost on test", AMBER)
        metric_card(c4, f"{cs['top10_list_size']:,}", "Win-back target list", f"{cs['top10_precision']:.0%} actually churn", GREEN)
        interp(f"Production uses the top-10% list, not the 0.5 cutoff. The list of "
               f"{cs['top10_list_size']:,} highest-risk customers contains "
               f"{cs['top10_churners_caught']:,} real churners, so contacting it is far more "
               f"efficient than blanket outreach.")
        bench = A["churn_metrics_df"].copy()
        bench["Lift@10%"] = (bench["Prec@10%"] / BASE_RATE).round(1)
        bench.loc["Baseline (no model)"] = {"Accuracy": round(1 - BASE_RATE, 3), "Precision": round(1 - BASE_RATE, 3),
                                            "Recall": 0.0, "F1-Score": 0.0, "ROC-AUC": 0.5,
                                            "Prec@10%": round(BASE_RATE, 3), "Lift@10%": 1.0}
        tag("FULL ML METRICS (TEMPORAL TEST PERIOD)")
        st.dataframe(bench.style.highlight_max(axis=0), width='stretch')
        interp("All metrics are measured on the unseen test period. The baseline always predicts "
               "return, so it catches no churners. Lift at 10% compares the top decile with the "
               "base rate. The Voting ensemble reaches about 4.95x.")
        cm = voting_test_confusion()
        tn, fp, fn, tp = cm.ravel()
        c1, c2 = st.columns(2)
        c1.markdown("**Confusion matrix (Voting, temporal test)**")
        c1.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  template="plotly_white", labels=dict(x="Predicted", y="Actual"),
                                  x=["Returned", "Churned"], y=["Returned", "Churned"]), width='stretch')
        c2.markdown("**How to read it**")
        c2.markdown(f"- True negatives {tn:,} - returned and predicted returned")
        c2.markdown(f"- False positives {fp:,} - returned but flagged churned")
        c2.markdown(f"- False negatives {fn:,} - churned but predicted returned (the miss)")
        c2.markdown(f"- True positives {tp:,} - churned and predicted churned")
        interp(f"At the 0.5 threshold the model catches {tp:,} of {tp + fn:,} churners. Recall is "
               f"modest because churn is rare (9.6 percent). Production uses the top-decile "
               f"ranking instead, where about {cs['top10_precision']:.0%} of the targeted customers actually churn.")
        bv = train_test_auc()
        st.markdown("**Bias-variance view (train vs test AUC)**")
        st.dataframe(bv, width='stretch')
        interp("The gap between train and test AUC is the overfit signal. Logistic has almost "
               "no gap with near-equal test AUC, and the Voting ensemble keeps the best test AUC "
               "with a moderate gap. Together they cover the bias-variance trade-off.")

        tag("MODEL EQUATION AND FEATURE THRESHOLDS")
        st.markdown("**Churn probability from the logistic model**")
        st.latex(r"P(\text{churn}) = \frac{1}{1 + \exp\big(-(\beta_0 + \sum_i \beta_i x_i)\big)}")
        lr = A["fitted_models"]["Logistic (Ch4)"]
        coef = pd.Series(lr.coef_[0], index=A["feature_names"]).sort_values(ascending=False)
        topc = pd.concat([coef.head(5), coef.tail(3)]).rename("beta")
        st.dataframe(topc.round(3).to_frame(), width='stretch')
        st.caption(f"Intercept beta0 = {lr.intercept_[0]:.3f} on standardized features. Positive "
                   f"beta raises churn odds, negative beta protects.")
        st.markdown("**Feature thresholds from a shallow decision tree (raw units, "
                    "interpretability only)**")
        num = tr[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
        Xraw = tr[num].fillna(tr[num].median())
        dt_raw = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE).fit(Xraw, y_tr)
        tree_ = dt_raw.tree_
        rules = []

        def walk(node, depth, cond):
            if tree_.children_left[node] == -1:
                p = tree_.value[node][0][1] / tree_.value[node][0].sum()
                rules.append((cond or "all customers", p, int(tree_.n_node_samples[node])))
                return
            if depth >= 2:
                return
            f = tree_.feature[node]
            th = tree_.threshold[node]
            walk(tree_.children_left[node], depth + 1, cond + (f"{num[f]} <= {th:.1f}" if not cond else f" and {num[f]} <= {th:.1f}"))
            walk(tree_.children_right[node], depth + 1, cond + (f"{num[f]} > {th:.1f}" if not cond else f" and {num[f]} > {th:.1f}"))

        walk(0, 0, "")
        for cond, p, n in rules:
            st.markdown(f"- IF {cond} THEN churn probability {p:.0%} (n = {n:,})")
        st.caption("Each rule shows a real threshold in real units. For example, a long recency "
                   "with low order count marks the high-churn branch of the tree.")

        tag("FEATURES USED AND IMPORTANCE BY MODEL")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**All {len(FEATURES)} features** (behaviour, category, demographics; race/ethnicity excluded)")
            groups = {
                "Behavioural": ["orders_12m", "spend_12m", "aov_12m", "aov_recent", "aov_prior", "aov_trend",
                                "recency_days", "active_months_12m", "category_count_12m", "avg_price_12m", "gift_card_share"],
                "Category": ["entertainment_share", "gaming_share", "media_share", "ott_flag"],
                "Demographics": ["Age", "Annual_Income_USD", "Household_Size", "Reported_Usage_Freq",
                                 "Gender", "Education_Level", "Region"]}
            for g, fs in groups.items():
                st.markdown(f"**{g}** - " + ", ".join(fs))
            st.caption("The same feature set feeds every model. Features are standardized and "
                       "one-hot encoded before training.")
        with c2:
            st.markdown("**Top features by model**")
            imp = per_model_importance()
            model_choice = st.selectbox("Model", list(imp.keys()) + ["Voting (Ch7)"], index=0)
            if model_choice in imp:
                imp_s = imp[model_choice]
                fig = px.bar(x=imp_s.values, y=imp_s.index, orientation="h", template="plotly_white",
                             labels={"x": "Importance", "y": ""}, color=imp_s.values,
                             color_continuous_scale="Blues").update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, width='stretch')
            else:
                st.markdown("Voting averages the probabilities of Logistic, Random Forest and "
                            "Gradient Boosting, so its importance is the union of its members. "
                            "Select a member model to see its features.")
            interp("Recency, activity and order velocity dominate every model. Demographics "
                   "matter little. The custom-training lab shows the same view for the model "
                   "you train in tab 3.")

# ================================================================ 4
with T[IDX["4 · Importance"]]:
    doing("The drivers of churn are identified from three independent methods, and the "
          "stability of the churn definition is verified.")

    c1, c2 = st.columns(2)
    with c1:
        tag("SPLIT IMPORTANCE")
        fig = px.bar(A["rf_imp"].head(10), x="MDI_Importance", y="Feature", orientation="h",
                     template="plotly_white", color="MDI_Importance", color_continuous_scale="Blues",
                     labels={"MDI_Importance": "Importance"}).update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, width='stretch')
        interp("How strongly each feature splits churners from stayers.")
    with c2:
        tag("PERMUTATION IMPORTANCE")
        fig = px.bar(A["perm_imp"].head(10), x="Perm_Importance", y="Feature", orientation="h",
                     template="plotly_white", color="Perm_Importance", color_continuous_scale="Viridis",
                     labels={"Perm_Importance": "Importance"}).update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, width='stretch')
        interp("How much accuracy drops when a feature is shuffled. This shows what the model "
               "truly needs.")

    tag("SHAP VALUES (XGBOOST - MODERN EXPLAINABILITY)")
    sh = A["shap_importance"].head(12)
    fig = px.bar(sh, x="Mean_Abs_SHAP", y="Feature", orientation="h", template="plotly_white",
                 color="Mean_Abs_SHAP", color_continuous_scale="Viridis",
                 labels={"Mean_Abs_SHAP": "Mean |SHAP|"}).update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig, width='stretch')
    interp("SHAP gives every feature a per-customer contribution to the churn score. It agrees "
           "with MDI and permutation importance, confirming the drivers are stable.")

    tag("DRIVERS WITH DIRECTION (LOGISTIC ODDS)")
    lr = A["fitted_models"]["Logistic (Ch4)"]
    coef = pd.Series(lr.coef_[0], index=A["feature_names"]).sort_values()
    drv = pd.concat([coef.head(6), coef.tail(6)]).rename("coef").to_frame()
    drv["dir"] = np.where(drv["coef"] > 0, "Increases churn risk", "Protects against churn")
    fig = px.bar(drv, x="coef", y=drv.index, orientation="h", template="plotly_white",
                 labels={"coef": "Log-odds change per 1 std unit", "y": ""},
                 color="dir", color_discrete_map={"Increases churn risk": RED, "Protects against churn": GREEN}
                 ).update_layout(yaxis={"autorange": "reversed"}, showlegend=True)
    st.plotly_chart(fig, width='stretch')
    interp("Recency and a falling order count raise churn odds. Order velocity, active months "
           "and category breadth lower them. Demographics have small effects. Correlated "
           "features are checked with VIF in the notebook.")

    tag("DEFINITION ROBUSTNESS (ML METRICS)")
    sdf = A["sensitivity_df"].copy()
    sdf.columns = ["Churn window (days)", "Base rate (train)", "Base rate (test)", "AUC (temporal test)"]
    st.dataframe(sdf, width='stretch')
    interp("AUC stays within 0.87-0.90 across 30-180 day definitions. The model is robust to "
           "the churn cutoff, so 90 days is a defensible choice.")

# ================================================================ 5 (survival, optional)
if "Survival" in scope:
    with T[IDX["5 · Survival"]]:
        doing("Time to first purchase (time-to-return) is modelled with survival "
              "analysis. S(t) is the probability that a customer has NOT returned by day "
              "t - i.e. a probability of no-return - so S(90) is the 90-day probability "
              "of no return. Censored customers, who never returned before the data end, "
              "are handled explicitly. NOTE: train/test snapshots share the same cohort "
              "and outcome windows overlap, so these results are in-sample only.")
        sv = A["survival"]
        sp = sv["probs"]
        c1, c2 = st.columns(2)
        c1.metric("C-index (RSF)", f'{sv["cindex_rsf"]:.3f}',
                  help="0.5 is random ranking, 1.0 is perfect. How well the survival model "
                       "ranks time to return.")
        c2.metric("C-index (Cox)", f'{sv["cindex_cox"]:.3f}',
                  help="Cox proportional-hazards candidate only as baseline; no PH "
                       "assumption tested.")
        c1, c2 = st.columns(2)
        with c1:
            tag("MEAN LAPSE PROBABILITY BY HORIZON")
            means = sp[[f"S_{h}" for h in sv["horizons"]]].mean()
            fig = px.line(x=[f"{h}d" for h in sv["horizons"]], y=means.values, markers=True,
                          template="plotly_white", labels={"x": "Horizon", "y": "Mean S(t)"})
            fig.update_traces(line_color=BLUE, marker_color=BLUE)
            st.plotly_chart(fig, width='stretch')
            interp("S(90) is about 0.099, matching the binary 90-day churn rate of 9.6 percent. The "
                   "two views agree, and the survival curve adds the time dimension (its "
                   "c-index is in-sample only, see the caveat above).")
        with c2:
            tag("EXPECTED DAYS TO RETURN (URGENCY)")
            fig = px.histogram(sp, x="Expected_Days_To_Return", nbins=30, template="plotly_white",
                               color_discrete_sequence=[BLUE])
            st.plotly_chart(fig, width='stretch')
            interp("A short expected return time means the customer is urgent but less likely "
                   "lost. The optimizer weights offers by S(30) to prioritise those closest to "
                   "no-return (never returning).")
        tag("SURVIVAL SCORES (SAMPLE)")
        st.dataframe(sp.head(8).round(3), width='stretch')
        interp(f"Correlation between survival S(90) and the binary churn probability is "
               f"{sv['corr_s90_binary']:.2f}. Survival adds urgency, which the binary model "
               f"cannot express, and the same MILP optimizer accepts both probability sources.")

# ================================================================ 6 (segments)
with T[IDX["6 · Segments"]]:
    doing("Natural customer groups are derived from behaviour via principal components and "
          "K-Means, then named from their actual profiles.")
    k_choice = st.selectbox("Number of segments (k)", [3, 4, 5, 6], index=1,
                            help="4 is validated with silhouette near 0.22. Higher k gives finer "
                                 "groups but softer boundaries.")

    seg_feats = ["Total_Spend_USD", "Order_Count", "Avg_Order_Value_USD", "Orders_Per_Year",
                 "Active_Months", "Category_Count", "Recency_Days", "entertainment_share", "ott_flag",
                 "Age", "Annual_Income_USD", "Household_Size"]
    Xs = StandardScaler().fit_transform(cust[seg_feats].fillna(cust[seg_feats].median()))
    pca6_model = PCA(n_components=6, random_state=RANDOM_STATE)
    pca6 = pca6_model.fit_transform(Xs)

    tag("PCA VARIANCE AND SILHOUETTE")
    c1, c2 = st.columns(2)
    with c1:
        pca_all = PCA(random_state=RANDOM_STATE).fit(Xs)
        cum = np.cumsum(pca_all.explained_variance_ratio_)
        n95 = int(np.argmax(cum >= 0.95)) + 1
        fig = px.line(x=range(1, len(cum) + 1), y=cum, markers=True, template="plotly_white",
                      labels={"x": "Component", "y": "Cumulative variance"})
        fig.add_hline(y=0.95, line_dash="dash", line_color=RED)
        st.plotly_chart(fig, width='stretch')
        load = pd.DataFrame(pca_all.components_[:2], columns=seg_feats, index=["PC1", "PC2"])
        for pc in ["PC1", "PC2"]:
            top = load.loc[pc].abs().sort_values(ascending=False).head(4)
            st.caption(f"{pc} axis - " + ", ".join(f"{f} {load.loc[pc, f]:+.2f}" for f in top.index))
    with c2:
        sils = silhouette_by_k(pca6)
        fig = px.line(x=list(sils.keys()), y=list(sils.values()), markers=True, template="plotly_white",
                      labels={"x": "Segments (k)", "y": "Silhouette"})
        st.plotly_chart(fig, width='stretch')
    interp(f"Six components retain {pca6_model.explained_variance_ratio_.sum():.0%} of variance "
           f"and silhouettes are comparable across k=3-7, so k=4 is chosen for business "
           f"interpretability. PC1 separates value and activity, PC2 separates engagement breadth.")

    km = KMeans(n_clusters=k_choice, random_state=RANDOM_STATE, n_init=10).fit(pca6)
    prof = cust.assign(Cluster=km.labels_).groupby("Cluster").agg(
        n=("Customer_ID", "count"), CLV=("CLV_3Yr", "mean"), Spend=("Total_Spend_USD", "mean"),
        Freq=("Orders_Per_Year", "mean"), ActiveM=("Active_Months", "mean"),
        Ent=("entertainment_share", "mean"), OTT=("ott_flag", "mean"), Recency=("Recency_Days", "mean")).round(2)
    order = prof.sort_values("CLV", ascending=False).index
    tiers = ["High-Value Loyalists", "Established Shoppers", "Occasional Shoppers", "Light / Disengaged",
             "Casual / New", "Inactive"]
    seg_map = {idx: tiers[i] if i < len(tiers) else f"Segment {idx}" for i, idx in enumerate(order)}
    prof = prof.rename(index=seg_map).sort_values("CLV", ascending=False)

    c1, c2 = st.columns(2)
    c1.metric("Segments", f"k = {k_choice}",
              f"silhouette {__import__('sklearn.metrics', fromlist=['silhouette_score']).silhouette_score(pca6, km.labels_):.3f}")
    c2.metric("Total 3-yr value pool", f"${cust['CLV_3Yr'].sum():,.0f}")
    st.dataframe(prof, width='stretch')
    interp("High-Value Loyalists are few and worth the most. Light / Disengaged is the largest "
           "risk pool. Occasional shoppers are the cross-sell opportunity.")

    overlay = (cust.assign(Cluster=km.labels_)[["Customer_ID", "Cluster"]]
               .merge(A["test_scores"][["Customer_ID", "Churn_Prob", "CLV_3Yr"]], on="Customer_ID", how="inner")
               .assign(Segment=lambda d: d["Cluster"].map(seg_map)))
    sc = overlay.groupby("Segment").agg(n=("Customer_ID", "count"),
                                        avg_churn_prob=("Churn_Prob", "mean")).round(3)
    sc["Value at risk (3yr $)"] = overlay.groupby("Segment")["Churn_Prob"].apply(
        lambda s: (s * overlay.loc[s.index, "CLV_3Yr"].fillna(0)).sum()).round(0)
    st.markdown("**Churn risk and value at risk by segment**")
    st.dataframe(sc, width='stretch')
    interp("Value at risk is churn probability times CLV, summed per segment. It is the dollar "
           "exposure each segment represents.")

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    viz = pd.DataFrame({"PC1": pca2[:, 0], "PC2": pca2[:, 1],
                        "Segment": cust.assign(Cluster=km.labels_)["Cluster"].map(seg_map),
                        "CLV": cust["CLV_3Yr"]})
    st.plotly_chart(px.scatter(viz, x="PC1", y="PC2", color="Segment", size="CLV", opacity=0.6,
                               template="plotly_white", size_max=26), width='stretch')

# ================================================================ 7 (optimizer)
with T[IDX["7 · Optimizer (MILP)"]]:
    doing("The retention budget is allocated exactly with a Mixed-Integer Linear Program "
          "solved by HiGHS. Customers and offer sizes are chosen so expected retained value "
          "is maximized under the budget.")
    if "Survival" in scope:
        use_surv = st.checkbox("Weight offers by time-to-return (survival)",
                               value=False,
                               help="Uses the survival-based 90-day no-return probability S(90) and an "
                                     "urgency weight S(30) inside the objective. Off uses the binary "
                                     "model probability. Survival weighting is solved only when you "
                                     "ask for it, so the first page load stays cheap.")
    else:
        use_surv = False
    B = st.slider("Retention budget ($)", 1000, 20000, 10000, 1000, key="milp_budget",
                  help="The exact MILP optimum is computed at the chosen budget.")

    ts_off = A["test_scores"].copy()
    if use_surv:
        sv = A["survival"]
        ts_off = ts_off.merge(sv["probs"][["Customer_ID", "S_30", "S_90"]], on="Customer_ID", how="left")
        ts_off["Churn_Prob"] = ts_off["S_90"].fillna(ts_off["Churn_Prob"])
        ts_off["urgency"] = ts_off["S_30"].fillna(1.0)
    else:
        ts_off["urgency"] = 1.0

    def response(d, base=0.04, slope=0.018, cap=0.60):
        return float(np.clip(base + slope * d, 0.0, cap))

    precomputed = OFFER["milp_by_budget"][int(B)]
    if use_surv:
        solved = st.session_state.setdefault("milp_surv", {})
        if st.button("Solve survival-weighted allocation", type="primary",
                     help="Runs the exact MILP once at the selected budget and keeps the result."):
            with st.spinner("Solving the exact MILP with survival weights..."):
                solved[int(B)] = solve_milp_live(ts_off, B)
        m = solved.get(int(B))
        if m is None:
            st.info("Click the button above to solve the survival-weighted allocation at this "
                    "budget. Until then the cards show the precomputed binary-probability "
                    "optimum for the same budget.")
            m = precomputed
    else:
        m = precomputed

    # greedy + naive for the selected mode (fast, live)
    _p = ts_off["Churn_Prob"].to_numpy()
    _V = ts_off["CLV_3Yr"].to_numpy()
    _U = ts_off["urgency"].to_numpy()
    _d = np.array([5, 10, 15, 20, 25], dtype=float)
    _resp = np.clip(0.04 + 0.018 * _d, 0.0, 0.60) - 0.04
    _incr = (_p * _V * _U)[:, None] * _resp[None, :]
    _ratio = _incr / _d[None, :]
    _mask_any = _incr > 0
    _ratio = np.where(_mask_any, _ratio, 0.0)
    _best_k = _ratio.argmax(axis=1)
    _row_idx = np.arange(len(_p))
    _best_offer = np.where(_ratio[_row_idx, _best_k] > 0, _d[_best_k], 0.0)
    _best_incr = np.where(_ratio[_row_idx, _best_k] > 0, _incr[_row_idx, _best_k], 0.0)
    gdf = pd.DataFrame({"Offer_USD": _best_offer, "Incr_Retained_Rev": _best_incr})
    gdf = gdf[gdf.Offer_USD > 0].sort_values("Incr_Retained_Rev", ascending=False)
    gc = gdf.Offer_USD.cumsum().values
    gr = gdf.Incr_Retained_Rev.cumsum().values
    rc = gdf.sort_values("Offer_USD").Offer_USD.cumsum().values
    rr = gdf.sort_values("Offer_USD").Incr_Retained_Rev.cumsum().values
    kg = int(np.searchsorted(gc, B, side="right"))
    kn = int(np.searchsorted(rc, B, side="right"))
    greedy_rev = gr[kg - 1] if kg > 0 else 0.0
    naive_rev = rr[kn - 1] if kn > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"${m['rev']:,.0f}", "Retained value (MILP)", f"ROAS {m['roas']:.1f}x", GREEN)
    metric_card(c2, f"${greedy_rev:,.0f}", "Greedy heuristic", "value per dollar ordering", AMBER)
    metric_card(c3, f"${naive_rev:,.0f}", "Naive spend", "cheapest first", RED)
    metric_card(c4, f"{m['customers']:,}", "Customers offered", "at the optimum", BLUE)
    interp(f"The optimum protects ${m['rev']:,.0f}, which is {m['roas']:.1f}x per dollar, versus "
           f"${greedy_rev:,.0f} for the greedy and ${naive_rev:,.0f} for naive spend. "
           + ("Survival weighting is active, so high-value customers with the highest "
               "no-return probability are prioritised." if use_surv else "The binary-model probability is used."))
    st.caption("These figures are read from the precomputed artifact (train_models.py), using "
               "voting-ensemble probabilities - the same scoring as the executed notebook, so at "
               "a $10,000 budget both report the identical figure (about $159,027 retained, ROAS "
               "15.9x, greedy $66,664, naive $57,768). Survival weighting is solved on request. "
               "ROAS is a projection of the assumed response curve, not a measured return.")

    cv = OFFER["curve"]
    fig = px.line(cv, x="Budget_USD", y="Retained_Rev_MILP", markers=True, template="plotly_white",
                  labels={"Budget_USD": "Budget ($)", "Retained_Rev_MILP": "Retained value ($)"},
                  title="Retained value vs budget (binary-probability baseline)")
    fig.update_traces(line_color=GREEN, marker_color=GREEN, name="MILP optimum (exact)")
    fig.add_scatter(x=cv["Budget_USD"], y=cv["Retained_Rev_Greedy"], mode="lines+markers",
                    name="Greedy heuristic", line_color=AMBER, marker_color=AMBER)
    fig.add_scatter(x=cv["Budget_USD"], y=cv["Retained_Rev_CheapestFirst"], mode="lines+markers",
                    name="Naive (cheapest first)", line_color=RED, marker_color=RED)
    fig.add_vline(x=B, line_dash="dash", line_color=NAVY)
    st.plotly_chart(fig, width='stretch')
    interp("The gap between the curves is the dollar value of exact optimization. The curve "
           "uses the binary probability; the cards and list use the selected scoring mode.")

    with st.expander("Optimization mathematics"):
        st.markdown("**Objective and constraints** (0/1-knapsack. $x_{i,d}$ = 1 if customer $i$ "
                    "receives offer $d$)")
        st.latex(r"\max_{x} \; \sum_{i} \sum_{d \in D} \Delta V_i(d)\, x_{i,d}")
        st.latex(r"\text{s.t.} \; \sum_{d} x_{i,d} \le 1 \; \forall i, \quad "
                 r"\sum_{i} \sum_{d} d\, x_{i,d} \le B, \quad x_{i,d} \in \{0,1\}")
        st.markdown("**Expected retained value and the response assumption**")
        st.latex(r"\Delta V_i(d) = p_i \cdot V_i \cdot u_i \cdot (r(d) - r(0)), \qquad "
                 r"r(d) = \min(0.60,\; 0.04 + 0.018\,d)")
        st.caption("p_i is the churn probability (binary or survival S(90)), u_i is the urgency "
                   "weight S(30) when survival mode is on. Solved with HiGHS via "
                   "scipy.optimize.milp. The response curve is an assumption to be validated by "
                   "A/B test.")

    st.markdown("**Priority win-back list (exact MILP allocation)**")
    shown = m["allocation"].head(m["customers"])
    st.dataframe(shown[["Customer_ID", "Churn_Prob", "CLV_3Yr", "Offer_USD", "Incr_Retained_Rev"]]
                 .rename(columns={"Incr_Retained_Rev": "Expected retained $"}).reset_index(drop=True),
                 width='stretch', height=320)
    interp("These customers are contacted first. The table shows risk, value at stake, and the "
           "optimal offer size. Larger offers are concentrated on high-value high-risk "
           "customers" + (", prioritised by no-return urgency (survival mode)." if use_surv else "."))
    st.download_button("Download full MILP allocation (CSV)", m["allocation"].to_csv(index=False).encode(),
                       file_name="milp_allocation.csv", mime="text/csv")

# ================================================================ 8 (A/B test)
with T[IDX["8 · A/B Test"]]:
    doing("The win-back experiment is designed with a two-proportion z-test. Sample size and "
          "power are computed from the hypotheses, and the mathematics is shown below.")
    c1, c2, c3 = st.columns(3)
    p0 = c1.slider("Baseline 90-day repurchase rate", 0.30, 0.95, 0.552, 0.01,
                   help="0.552 is the top-decile high-risk cohort. 0.90 is all customers.")
    mde = c2.slider("Minimum detectable effect (pp)", 2, 20, 10, 1)
    alpha = c3.slider("Significance level alpha", 0.01, 0.10, 0.05, 0.01)
    c4, c5 = st.columns(2)
    pwr_target = c4.slider("Power target", 0.50, 0.95, 0.80, 0.05)
    n_avail = c5.slider("Customers per group available", 50, 800, 240, 10,
                        help="240 is half of the 480-customer top-decile list.")

    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(pwr_target)
    p1 = p0 + mde / 100
    n_req = int(np.ceil(((za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1))) / (p1 - p0) ** 2))
    pow_avail = stats.norm.cdf(np.sqrt(n_avail * (p1 - p0) ** 2 / (p0 * (1 - p0) + p1 * (1 - p1))) - za)

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, f"{n_req:,}", "Customers per group needed", f"{2 * n_req:,} total at {pwr_target:.0%} power", NAVY)
    metric_card(c2, f"{pow_avail:.0%}", "Power at available sample", f"{n_avail:,} per group", GREEN if pow_avail >= pwr_target else AMBER)
    metric_card(c3, f"+{mde}pp", "Minimum detectable effect", f"from {p0:.0%} baseline", BLUE)
    metric_card(c4, f"{p1 - p0:+.0%}", "Effect under H1", f"p1 = {p1:.1%}", GRAY)
    interp(f"At {n_avail:,} customers per group the design detects a +{mde}pp lift with "
           f"{pow_avail:.0%} power. The current top-decile list, 240 per group, reaches the "
           f"power target only for lifts around +12pp, so the rollout may need the top three "
           f"deciles, about 1,440 customers, or a directional read.")

    mdes = np.arange(0.02, 0.20, 0.005)
    pow_curve = [stats.norm.cdf(np.sqrt(n_avail * d ** 2 / (p0 * (1 - p0) + (p0 + d) * (1 - p0 - d))) - za) for d in mdes]
    fig = px.line(x=mdes * 100, y=pow_curve, markers=True, template="plotly_white",
                  labels={"x": "Minimum detectable effect (pp)", "y": "Power"})
    fig.update_traces(line_color=BLUE, marker_color=BLUE, name="Power at available sample")
    fig.add_hline(y=pwr_target, line_dash="dash", line_color=RED)
    st.plotly_chart(fig, width='stretch')
    interp("The curve shows the detectable effect at the available sample size. The red line "
           "is the power target.")

    n_curve = [(d, int(np.ceil(((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(pwr_target)) ** 2
                  * (p0 * (1 - p0) + (p0 + d) * (1 - p0 - d))) / d ** 2))) for d in mdes]
    fig2 = px.line(x=[d * 100 for d, _ in n_curve], y=[n for _, n in n_curve], markers=True,
                   template="plotly_white", labels={"x": "Minimum detectable effect (pp)", "y": "Customers per group"})
    fig2.update_traces(line_color=GREEN, marker_color=GREEN)
    fig2.add_hline(y=n_avail, line_dash="dash", line_color=NAVY)
    st.plotly_chart(fig2, width='stretch')
    interp("The required sample falls steeply as the detectable effect grows. The navy line is "
           "the available sample.")

    with st.expander("A/B test mathematics"):
        st.markdown("**Hypotheses** (one-sided, outcome is the 90-day repurchase rate p)")
        st.latex(r"H_0: p_T - p_C \le 0 \qquad H_1: p_T - p_C = \delta > 0")
        st.markdown("**Sample size per group** (two-proportion z-test)")
        st.latex(r"n = \frac{(z_{\alpha/2} + z_\beta)^2\,(p_0(1-p_0) + p_1(1-p_1))}{(p_1 - p_0)^2}")
        st.markdown("**Power at a given sample size**")
        st.latex(r"\Phi\Big(\frac{\sqrt{n}\,(p_1 - p_0)}{\sqrt{p_0(1-p_0) + p_1(1-p_1)}} - z_{\alpha/2}\Big)")
        st.markdown("**Test statistic** (observed, after the experiment)")
        st.latex(r"z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}_{pool}(1-\hat{p}_{pool})\left(\frac{1}{n_T} + \frac{1}{n_C}\right)}}, \qquad \hat{p}_{pool} = \frac{x_T + x_C}{n_T + n_C}")
        st.markdown("**Confidence interval** (two-sided, for the true difference)")
        st.latex(r"(\hat{p}_T - \hat{p}_C) \pm z_{\alpha/2}\,\sqrt{\frac{\hat{p}_T(1-\hat{p}_T)}{n_T} + \frac{\hat{p}_C(1-\hat{p}_C)}{n_C}}")
        st.caption("x is the number of repurchases and n is the group size. The worked example in "
                   "the notebook (z = 2.93, 95 percent interval of plus 3.5 to 17.5 percentage "
                   "points for a 10.5 point readout) is an ILLUSTRATIVE power example with "
                   "synthetic numbers - not an executed experiment. The design is 50/50 "
                   "stratified by score decile with a fixed 90-day readout. Guardrails are "
                   "support tickets, return rate and average spend.")

# ================================================================ 9 (recommendations)
with T[IDX["9 · Recommendations"]]:
    doing("Actions are prioritized for the business, and the methodology and limitations are "
          "documented for reviewers.")

    tag("RECOMMENDATIONS")
    recs = [
        ("P1", "Deploy the weekly churn score in the CRM",
         f"the top-10% list ({N_AT_RISK:,} customers, ${REV_AT_RISK:,.0f} at risk) is routed to "
         f"the win-back workflow. About 49% would churn without intervention."),
        ("P1", "Replace blanket discounts with the MILP allocation",
         f"at a ${OFFER['budget']:,.0f} budget the exact allocation retains "
         f"${OFFER['opt_rev']:,.0f} with ROAS {OFFER['roas']:.1f}x, which is "
         f"${OFFER['opt_rev']-OFFER['rand_rev']:,.0f} more than naive spend."),
        ("P2", "Tier the win-back by segment",
         "premium offers for at-risk High-Value Loyalists, low-cost coupons for Light / Disengaged, "
         "and category cross-sell for Occasional shoppers."),
        ("P2", "Add CX early-warning triggers",
         "a declining average order value or recency above 60 days flags a customer weeks before "
         "the 90-day silence."),
        ("P3", "Upgrade to survival-based urgency",
         "use S(30) and S(90) from the survival model to weight offers by time-to-return, so "
         "high-value customers with the highest no-return probability are contacted first."),
        ("P3", "A/B test the response curve, then automate",
         "randomize 50/50 within the top-decile list, measure 90-day repurchase lift, then run a "
         "rolling monthly retrain loop."),
    ]
    for p, t, d in recs:
        color = {"P1": RED, "P2": AMBER, "P3": GRAY}[p]
        st.markdown(f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;'
                    f'padding:8px 12px;margin:5px 0;font-size:.95rem;">'
                    f'<b style="color:{color}">{p}</b> &nbsp;<b>{t}</b> - {d}</div>', unsafe_allow_html=True)

    tag("ANALYTICAL RIGOR")
    st.markdown("- Temporal validation. The model is evaluated on a future period it never saw, "
                "with no censored outcomes.")
    st.markdown("- No leakage. Features use only pre-cutoff orders and the churn window is "
                "sensitivity-tested (AUC 0.87-0.90).")
    st.markdown("- Multicollinearity is checked with VIF; drivers are confirmed by MDI, "
                "permutation, SHAP and logistic odds.")
    st.markdown("- Survival analysis handles censoring and adds time-to-return urgency (the "
                "c-index is in-sample only - train/test cohort and outcome windows overlap).")
    st.markdown("- The greedy heuristic was measured about 60% below the MILP optimum in the "
                "executed notebook. The exact solver is used.")
    st.markdown("- The offer-response curve is an assumption. The A/B test is the causal upgrade.")

    tag("LIMITATIONS")
    st.markdown("- The panel is survey-recruited heavy users (median 221 orders). Rates may not "
                "generalize to all shoppers.")
    st.markdown("- CLV is a velocity proxy (annualized spend x 3) and a ranking measure.")
    st.markdown("- Survival horizons are capped by the data end (2023-03-19); long-horizon "
                "estimates are less certain.")
    st.markdown("- This is a single cross-section. Rolling retraining is the production design.")

    tag("GLOSSARY")
    for term, defn in [("Churn", "no purchase within 90 days."),
                       ("ROC-AUC", "how well churners are separated from stayers. 0.9 is excellent."),
                       ("Precision at top 10%", "share of contacted top-decile customers who actually churn. About 47%, or 4.9-5x the base rate."),
                       ("CLV (3-yr)", "annualized spend x 3. The dollars at stake."),
                       ("ROAS", "retained value per 1 dollar of offer spend. About 15.9x in the "
                                 "executed-notebook run - a projection of the assumed response "
                                 "curve, not a measured return."),
                       ("MILP", "Mixed-Integer Linear Program. Exact optimization via HiGHS."),
                       ("Survival S(t)", "probability a customer has NOT returned by day t "
                        "(no-return probability). S(90) is the 90-day no-return probability."),
                       ("C-index", "how well the survival model ranks time to return. 0.5 random, 1.0 perfect.")]:
        st.markdown(f"**{term}** - {defn}")

    st.markdown("---")
    st.caption("To reproduce run build_dataset.py, build_notebook.py, execute_notebook.py and "
               "train_models.py. Data is Open E-Commerce 1.0 (Berke et al. 2024, MIT, "
               "DOI 10.1038/s41597-024-03329-6). Notebook is amazon_churn_retention_analysis.ipynb")
