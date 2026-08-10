"""
Train churn models (temporal validation), segment customers, and optimize the
retention offer budget — Customer Churn Prediction & Retention Offer Optimization.

DATA (built by build_dataset.py):
  data/snapshot_train.csv - features measured up to 2022-03-19; label = no purchase in next 90 days
  data/snapshot_test.csv  - features measured up to 2022-09-19; same 90-day label
  data/customers.csv      - full-history customer table (CLV at risk, category mix)

METHOD:
  - TRUE temporal validation: train on period 1, test on period 2 (no leakage).
  - Churn models: Logistic, Decision Tree, Random Forest / Gradient Boosting /
    Voting. Metrics: AUC + precision@10% (business: targeting top decile).
  - Threshold sensitivity: same model retrained for 30/60/90/120/180-day churn windows.
  - Segmentation: PCA + K-Means on the full-history customer table.
  - Offer optimization: expected retained revenue = P(churn) x CLV-at-risk x
    response(offer); greedy allocation of a limited promo budget by marginal retained
    revenue per dollar; budget-vs-retained-revenue curve vs random targeting.

Usage:  python train_models.py
"""
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
from xgboost import XGBClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "models")

FEATURES = ["orders_12m", "spend_12m", "aov_12m", "aov_recent", "aov_prior", "aov_trend",
            "recency_days", "active_months_12m", "category_count_12m", "avg_price_12m",
            "entertainment_share", "gaming_share", "media_share", "gift_card_share", "ott_flag",
            "Age", "Annual_Income_USD", "Household_Size", "Hispanic", "Reported_Usage_Freq",
            "Gender", "Education_Level", "Race_Group", "Region"]

# Documented response assumption: probability a customer is retained given an offer of $d
# (small base return + linear response, saturating) - sensitivity-tested in the notebook.
def response(d, base=0.04, slope=0.018, cap=0.60):
    return float(np.clip(base + slope * d, 0.0, cap))


def load():
    tr = pd.read_csv(os.path.join(BASE, "data", "snapshot_train.csv"))
    te = pd.read_csv(os.path.join(BASE, "data", "snapshot_test.csv"))
    cust = pd.read_csv(os.path.join(BASE, "data", "customers.csv"))
    return tr, te, cust


def make_pre(df, feats):
    num = df[feats].select_dtypes(include=[np.number]).columns.tolist()
    cat = df[feats].select_dtypes(include=["object"]).columns.tolist()
    if not cat:
        pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        return pre.fit_transform(df[feats]), pre, num
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))]), cat)])
    Xp = pre.fit_transform(df[feats])
    names = num + list(pre.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat))
    return Xp, pre, names


def cls_metrics(y, p, prob):
    return {"Accuracy": round(accuracy_score(y, p), 4), "Precision": round(precision_score(y, p), 4),
            "Recall": round(recall_score(y, p), 4), "F1": round(f1_score(y, p), 4),
            "ROC-AUC": round(roc_auc_score(y, prob), 4)}


def precision_at_top_decile(y, prob):
    """% of the top 10% by churn probability who actually churned (targeting quality)."""
    k = max(1, len(y) // 10)
    top = np.argsort(-prob)[:k]
    return float(y.iloc[top].mean())


def build_artifacts():
    tr, te, cust = load()
    y_tr, y_te = tr["label_churn"], te["label_churn"]
    Xp_tr, pre, names = make_pre(tr, FEATURES)
    Xp_te = pre.transform(te[FEATURES])
    print(f"Train {Xp_tr.shape[0]:,} | Test {Xp_te.shape[0]:,} | features {Xp_tr.shape[1]} | "
          f"base rates {y_tr.mean():.1%} / {y_te.mean():.1%}")

    # ---------------- Churn models (Ch 4 / 6 / 7) ----------------------------
    models = {
        "Logistic (Ch4)": LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE).fit(Xp_tr, y_tr),
        "DecisionTree (Ch6)": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE).fit(Xp_tr, y_tr),
        "RandomForest (Ch7)": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1).fit(Xp_tr, y_tr),
        "GradientBoosting (Ch7)": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=RANDOM_STATE).fit(Xp_tr, y_tr),
        "XGBoost (Ch7)": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08,
                                       random_state=RANDOM_STATE, eval_metric="logloss").fit(Xp_tr, y_tr),
    }
    gb = models["GradientBoosting (Ch7)"]
    models["Voting (Ch7)"] = VotingClassifier(
        [("lr", models["Logistic (Ch4)"]), ("rf", models["RandomForest (Ch7)"]), ("gb", gb)], voting="soft").fit(Xp_tr, y_tr)

    rows, fitted = {}, {}
    for name, m in models.items():
        prob = m.predict_proba(Xp_te)[:, 1]
        p = (prob >= 0.5).astype(int)
        rows[name] = cls_metrics(y_te, p, prob)
        rows[name]["Prec@10%"] = round(precision_at_top_decile(y_te, prob), 4)
        fitted[name] = m
        print(f"[{name}] AUC {rows[name]['ROC-AUC']:.4f} | Prec@10% {rows[name]['Prec@10%']:.3f}")
    churn_metrics_df = pd.DataFrame(rows).T

    # Churn situation - concrete counts on the temporal test period (XGBoost)
    xgb_prob = fitted["XGBoost (Ch7)"].predict_proba(Xp_te)[:, 1]
    y_te_arr = np.asarray(y_te)
    topk = max(1, len(y_te) // 10)
    topk_prec = precision_at_top_decile(y_te, xgb_prob)
    churn_summary = {
        "test_customers": int(len(y_te)),
        "actual_churned": int(y_te_arr.sum()),
        "base_rate": round(float(y_te_arr.mean()), 4),
        "predicted_churned_05": int((xgb_prob >= 0.5).sum()),
        "top10_list_size": int(topk),
        "top10_precision": round(topk_prec, 4),
        "top10_churners_caught": int(round(topk_prec * topk)),
    }
    print("Churn summary (temporal test):", churn_summary)

    # SHAP importance for XGBoost (modern explainability)
    import shap
    explainer = shap.TreeExplainer(fitted["XGBoost (Ch7)"])
    sv = explainer.shap_values(Xp_te[:2000])
    shap_importance = pd.DataFrame({"Feature": names, "Mean_Abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("Mean_Abs_SHAP", ascending=False)

    # ---------------- Survival analysis (time-to-return; event = first purchase after cutoff) ------
    surv_tr = pd.read_csv(os.path.join(BASE, "data", "survival_train.csv"))
    surv_te = pd.read_csv(os.path.join(BASE, "data", "survival_test.csv"))
    s_tr = tr[["Customer_ID"]].merge(surv_tr, on="Customer_ID", how="left")   # keeps tr row order
    s_te = te[["Customer_ID"]].merge(surv_te, on="Customer_ID", how="left")

    def to_struct(d):
        return np.array([(bool(e), float(t)) for e, t in zip(d["event"], d["T_days"])],
                        dtype=[("event", "bool"), ("time", "float")])

    ys_tr, ys_te = to_struct(s_tr), to_struct(s_te)
    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=20, random_state=RANDOM_STATE, n_jobs=-1).fit(Xp_tr, ys_tr)
    cox = CoxPHSurvivalAnalysis().fit(Xp_tr, ys_tr)
    ci_r = concordance_index_censored(ys_te["event"], ys_te["time"], rsf.predict(Xp_te))[0]
    ci_c = concordance_index_censored(ys_te["event"], ys_te["time"], cox.predict(Xp_te))[0]
    # NOTE (in-sample caveat): the train/test snapshots share the same customer cohort and their
    # survival outcome windows overlap in calendar time (both end 2023-03-19), so the c-index below
    # is indicative only, NOT a clean temporal generalization of an unseen cohort.
    print(f"Survival c-index (test period; IN-SAMPLE - cohort/windows overlap with train) - RSF {ci_r:.4f} | Cox {ci_c:.4f}")

    horizons = [30, 60, 90, 120, 180]
    surv_fn = rsf.predict_survival_function(Xp_te)
    probs = np.array([[f(h) for h in horizons] for f in surv_fn])          # S(t) = P(no return by t)
    surv_probs = pd.DataFrame(probs, columns=[f"S_{h}" for h in horizons])
    surv_probs["Customer_ID"] = te["Customer_ID"].values
    surv_probs["Expected_Days_To_Return"] = np.trapezoid(probs, x=horizons)    # restricted mean
    surv_corr90 = float(np.corrcoef(probs[:, horizons.index(90)],
                                    fitted["XGBoost (Ch7)"].predict_proba(Xp_te)[:, 1])[0, 1])
    print(f"Survival S(90) vs binary XGBoost probability correlation: {surv_corr90:.3f}")
    survival = {"rsf": rsf, "cox": cox, "cindex_rsf": round(float(ci_r), 4),
                "cindex_cox": round(float(ci_c), 4), "horizons": horizons,
                "probs": surv_probs, "corr_s90_binary": round(surv_corr90, 4)}

    # ---------------- Threshold sensitivity (30/60/90/120/180 days) ----------
    print("\nThreshold sensitivity (GradientBoosting, temporal test):")
    sens = []
    for w in (30, 60, 90, 120, 180):
        mw = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4,
                                        random_state=RANDOM_STATE).fit(Xp_tr, tr[f"label_churn_{w}"])
        auc = roc_auc_score(te[f"label_churn_{w}"], mw.predict_proba(Xp_te)[:, 1])
        sens.append({"Window (days)": w, "Base rate (train)": round(tr[f"label_churn_{w}"].mean(), 3),
                     "Base rate (test)": round(te[f"label_churn_{w}"].mean(), 3), "AUC (temporal test)": round(auc, 4)})
        print(f"  {w:>3}d window -> AUC {auc:.4f}")
    sensitivity_df = pd.DataFrame(sens)

    # ---------------- Feature importance (Ch 7) ------------------------------
    rf = fitted["RandomForest (Ch7)"]
    rf_imp = pd.DataFrame({"Feature": names, "MDI_Importance": rf.feature_importances_}).sort_values("MDI_Importance", ascending=False)
    perm = permutation_importance(fitted["GradientBoosting (Ch7)"], Xp_te, y_te, n_repeats=5, random_state=RANDOM_STATE)
    perm_imp = pd.DataFrame({"Feature": names, "Perm_Importance": perm.importances_mean}).sort_values("Perm_Importance", ascending=False)

    # ---------------- Segmentation (Ch 8 / 9, full-history customers) --------
    seg_feats = ["Total_Spend_USD", "Order_Count", "Avg_Order_Value_USD", "Orders_Per_Year",
                 "Active_Months", "Category_Count", "Recency_Days", "entertainment_share", "ott_flag",
                 "Age", "Annual_Income_USD", "Household_Size"]
    Xs, pre_s, seg_names_full = make_pre(cust, seg_feats)
    pca6 = PCA(n_components=6, random_state=RANDOM_STATE).fit(Xs)
    Xp6 = pca6.transform(Xs)
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10).fit(Xp6)
    cust["Cluster"] = kmeans.labels_
    sil = silhouette_score(Xp6, kmeans.labels_)

    prof = cust.groupby("Cluster").agg(
        n=("Customer_ID", "count"), CLV=("CLV_3Yr", "mean"), Spend=("Total_Spend_USD", "mean"),
        Freq=("Orders_Per_Year", "mean"), ActiveM=("Active_Months", "mean"),
        Ent=("entertainment_share", "mean"), OTT=("ott_flag", "mean"),
        Recency=("Recency_Days", "mean"), Income=("Annual_Income_USD", "mean")).round(2)
    # auto-name by CLV (value tiers)
    order = prof.sort_values("CLV", ascending=False).index
    tier = ["High-Value Loyalists", "Established Shoppers", "Occasional Shoppers", "Light / Disengaged"]
    seg_names = {idx: tier[i] for i, idx in enumerate(order)}
    cust["Segment"] = cust["Cluster"].map(seg_names)
    segment_profiles = prof.rename(index=seg_names).sort_values("CLV", ascending=False)
    print(f"\nSegments (silhouette {sil:.3f}):"); print(segment_profiles.to_string())

    # ---------------- Test-period scores + offer optimization ----------------
    prob_te = fitted["GradientBoosting (Ch7)"].predict_proba(Xp_te)[:, 1]
    test_scores = te[["Customer_ID"]].copy()
    test_scores["Churn_Prob"] = np.round(prob_te, 4)
    test_scores = test_scores.merge(cust[["Customer_ID", "CLV_3Yr", "Segment"]], on="Customer_ID", how="left")
    test_scores = test_scores.dropna(subset=["CLV_3Yr"])

    OFFERS = [5, 10, 15, 20, 25]

    def greedy_alloc(ts, rfn, offers=OFFERS):
        rows_alloc = []
        for _, r in ts.iterrows():
            p, clv = r["Churn_Prob"], r["CLV_3Yr"]
            best_d, best_ratio = 0, 0.0
            for d in offers:
                incr = p * clv * (rfn(d) - rfn(0))
                if incr / d > best_ratio and incr > 0:
                    best_d, best_ratio = d, incr / d
            rows_alloc.append((r["Customer_ID"], p, clv, best_d,
                               p * clv * (rfn(best_d) - rfn(0)) if best_d else 0.0))
        al = pd.DataFrame(rows_alloc, columns=["Customer_ID", "Churn_Prob", "CLV_3Yr", "Offer_USD", "Incr_Retained_Rev"])
        return al[al["Offer_USD"] > 0].sort_values("Incr_Retained_Rev", ascending=False)

    def milp_alloc(ts, B, rfn, offers=OFFERS, time_limit=90):
        """Exact 0/1-knapsack budget allocation via scipy.optimize.milp (HiGHS)."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        p = ts["Churn_Prob"].values
        V = ts["CLV_3Yr"].values
        r0 = rfn(0)
        DV = np.array([[pi * Vi * (rfn(d) - r0) for d in offers] for pi, Vi in zip(p, V)])
        n = len(ts)
        A_bud = np.tile(np.array(offers, dtype=float), n)
        A_one = np.zeros((n, n * len(offers)))
        for i in range(n):
            A_one[i, i * len(offers):(i + 1) * len(offers)] = 1.0
        cons = [LinearConstraint(A_bud, ub=B), LinearConstraint(A_one, ub=1.0)]
        res = milp(c=-DV.ravel(), integrality=np.ones(n * len(offers)), bounds=Bounds(0, 1),
                   constraints=cons, options={"time_limit": time_limit})
        if not res.success:
            raise RuntimeError(f"MILP failed at B={B}: {res.message}")
        x = np.round(res.x).astype(int).reshape(n, len(offers))
        al = ts[["Customer_ID", "Churn_Prob", "CLV_3Yr"]].copy()
        al["Offer_USD"] = (x * np.array(offers)).sum(axis=1)
        al["Incr_Retained_Rev"] = (x * DV).sum(axis=1)
        al = al[al["Offer_USD"] > 0].sort_values("Incr_Retained_Rev", ascending=False).reset_index(drop=True)
        return {"allocation": al, "cost": float(al.Offer_USD.sum()), "rev": float(al.Incr_Retained_Rev.sum()),
                "customers": int(len(al)), "roas": float(al.Incr_Retained_Rev.sum() / max(al.Offer_USD.sum(), 1))}

    greedy = greedy_alloc(test_scores, response)
    budget_grid = np.arange(1000, 20001, 1000)

    # MILP (exact) solutions across the budget grid (headline method)
    milp_by_budget, curve = {}, []
    for B in budget_grid:
        m = milp_alloc(test_scores, B, response)
        milp_by_budget[int(B)] = m
        curve.append((B, m["rev"]))

    # Greedy + cheapest-first curves for comparison
    gc = greedy["Offer_USD"].cumsum().values
    gr = greedy["Incr_Retained_Rev"].cumsum().values
    rc = greedy.sort_values("Offer_USD")["Offer_USD"].cumsum().values
    rr = greedy.sort_values("Offer_USD")["Incr_Retained_Rev"].cumsum().values
    greedy_curve = [(B, gr[int(np.searchsorted(gc, B, side="right")) - 1] if np.searchsorted(gc, B, side="right") > 0 else 0.0) for B in budget_grid]
    naive_curve = [(B, rr[int(np.searchsorted(rc, B, side="right")) - 1] if np.searchsorted(rc, B, side="right") > 0 else 0.0) for B in budget_grid]
    budget_curve = pd.DataFrame({"Budget_USD": [c for c, _ in curve],
                                 "Retained_Rev_MILP": [v for _, v in curve],
                                 "Retained_Rev_Greedy": [v for _, v in greedy_curve],
                                 "Retained_Rev_CheapestFirst": [v for _, v in naive_curve]})

    B = 10000
    milp10 = milp_by_budget[B]
    k2 = int(np.searchsorted(rc, B, side="right"))
    naive_rev = rr[k2 - 1] if k2 > 0 else 0.0
    greedy_rev = float(gr[int(np.searchsorted(gc, B, side="right")) - 1]) if np.searchsorted(gc, B, side="right") > 0 else 0.0
    print(f"\nOffer optimization @ ${B:,.0f}: MILP (exact) retains ${milp10['rev']:,.0f} "
          f"(ROAS {milp10['roas']:.2f}x, {milp10['customers']:,} customers) | greedy ${greedy_rev:,.0f} "
          f"| naive ${naive_rev:,.0f} | MILP vs greedy gain {milp10['rev'] - greedy_rev:,.0f} "
          f"({(milp10['rev'] / max(greedy_rev, 1) - 1):.0%})")

    offer_result = {
        "budget": float(B), "opt_cost": milp10["cost"], "opt_rev": milp10["rev"],
        "rand_rev": naive_rev, "customers_offered": milp10["customers"], "roas": milp10["roas"],
        "greedy_rev": greedy_rev, "greedy_roas": float(greedy_rev / max(B, 1)),
        "gap_greedy_pct": float((1 - greedy_rev / milp10["rev"]) * 100),
        "allocation": milp10["allocation"], "curve": budget_curve,
        "milp_by_budget": milp_by_budget, "response_fn": "clip(0.04 + 0.018*d, 0, 0.60)",
    }

    artifacts = {
        "tr": tr, "te": te, "cust": cust, "feature_names": names,
        "pre": pre, "pre_s": pre_s, "pca6": pca6, "kmeans": kmeans,
        "churn_metrics_df": churn_metrics_df, "fitted_models": fitted,
        "churn_summary": churn_summary, "shap_importance": shap_importance,
        "survival": survival,
        "sensitivity_df": sensitivity_df,
        "rf_imp": rf_imp, "perm_imp": perm_imp,
        "segment_profiles": segment_profiles, "seg_names": seg_names, "silhouette": sil,
        "test_scores": test_scores, "prob_te": prob_te,
        "offer_result": offer_result,
    }
    return artifacts


def main():
    artifacts = build_artifacts()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    artifacts["test_scores"].to_csv(os.path.join(OUT, "churn_scores_test.csv"), index=False)
    artifacts["offer_result"]["allocation"].to_csv(os.path.join(OUT, "offer_allocation.csv"), index=False)
    print(f"\nSaved models/artifacts.pkl + churn_scores_test.csv + offer_allocation.csv "
          f"({os.path.getsize(os.path.join(OUT, 'artifacts.pkl'))/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
