"""
Build datasets for the Customer Churn Prediction & Retention Offer Optimization project.

INPUTS (real Open E-Commerce 1.0, Berke et al. 2024, MIT Media Lab):
  data/raw/ordered_open_ecommerce.parquet  - 1.76M Amazon purchases (2018-01 .. 2023-03)
  data/raw/survey.csv                      - 5,027 respondents demographics

OUTPUTS:
  data/customers.csv        - one row per customer: demographics + full-history
                              behavioural + category features + CLV-at-risk value
  data/snapshot_train.csv   - TEMPORAL train snapshot: features measured up to
                              cutoff 2022-03-19; label = churned if NO purchase in the
                              next 90 days (outcome window fully observed)
  data/snapshot_test.csv    - TEMPORAL test snapshot: features measured up to
                              cutoff 2022-09-19; same 90-day outcome definition.
                              (Train and test are DISJOINT IN TIME - no leakage.)

TEMPORAL DESIGN (the gold standard for churn):
   cutoff_train = 2022-03-19   -> outcome window [2022-03-19, 2022-06-17)
   cutoff_test  = 2022-09-19   -> outcome window [2022-09-19, 2022-12-18)
   Data ends 2023-03-19, so every outcome window is fully observed (no censoring).
   Features use ONLY orders before each cutoff - the model never sees the future.

CATEGORY GROUPS (for churn-by-category / OTT-style analysis):
   ENTERTAINMENT_DIGITAL: streaming subscriptions, downloadable games, music, movies,
                          TV series, video game software/hardware, e-reader content
   GAMING:                video-game categories (console, software, accessories)
   MEDIA_BOOKS:           ABIS_BOOK / e-books / music / physical movies & TV
   OTT_FLAG:              bought any STREAMING_* or KINDLE_UNLIMITED membership

Usage:  python build_dataset.py
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "data")

CUTOFF_TRAIN = pd.Timestamp("2022-03-19")
CUTOFF_TEST = pd.Timestamp("2022-09-19")
DATA_END = pd.Timestamp("2023-03-19")   # end of data collection
OUTCOME_DAYS = 90

# --- Demographics parsing (benign items only) ---------------------------------
AGE_MID = {"18 - 24 years": 21, "25 - 34 years": 29.5, "35 - 44 years": 39.5,
           "45 - 54 years": 49.5, "55 - 64 years": 59.5, "65 and older": 70}
INCOME_MID = {"Less than $25,000": 20000, "$25,000 - $49,999": 37500, "$50,000 - $74,999": 62500,
              "$75,000 - $99,999": 87500, "$100,000 - $149,999": 125000, "$150,000 or more": 175000}
EDU_MAP = {"Some high school or less": "High School or Less",
           "High school diploma or GED": "High School or Less",
           "Bachelor's degree": "Bachelor's",
           "Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS, etc)": "Graduate/Professional",
           "Prefer not to say": "Other"}
FREQ_MAP = {"Less than 5 times per month": 1, "5 - 10 times per month": 2, "More than 10 times per month": 3}
HH_MAP = {"1 (just me!)": 1, "2": 2, "3": 3, "4+": 4}
REGION = {"ME": "Northeast", "NH": "Northeast", "VT": "Northeast", "MA": "Northeast", "RI": "Northeast",
          "CT": "Northeast", "NY": "Northeast", "NJ": "Northeast", "PA": "Northeast",
          "OH": "Midwest", "MI": "Midwest", "IN": "Midwest", "IL": "Midwest", "WI": "Midwest",
          "MN": "Midwest", "IA": "Midwest", "MO": "Midwest", "ND": "Midwest", "SD": "Midwest",
          "NE": "Midwest", "KS": "Midwest",
          "DE": "South", "MD": "South", "DC": "South", "VA": "South", "WV": "South", "NC": "South",
          "SC": "South", "GA": "South", "FL": "South", "KY": "South", "TN": "South", "AL": "South",
          "MS": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
          "MT": "West", "ID": "West", "WY": "West", "CO": "West", "NM": "West", "AZ": "West",
          "UT": "West", "NV": "West", "WA": "West", "OR": "West", "CA": "West", "AK": "West", "HI": "West"}
STATE_ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
              "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
              "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
              "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
              "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
              "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
              "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
              "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
              "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
              "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}

# --- Category groups (regex patterns on the `category` column) -----------------
CAT = {
    "ENTERTAINMENT_DIGITAL": r"STREAMING|DOWNLOADABLE_VIDEO_GAME|PHYSICAL_VIDEO_GAME_SOFTWARE|VIDEO_GAME_CONSOLE|VIDEO_GAME_|CONSOLE_VIDEO_GAMES|PHYSICAL_MOVIE|PHYSICAL_TV_SERIES|ABIS_MUSIC|ABIS_VIDEO|ABIS_EBOOKS|AMAZON_BOOK_READER|PRELOADED_ELECTRONIC_GAME|MUSIC|DOWNLOADABLE_TV|SOFTWARE_GAMES|GAME_",
    "GAMING": r"VIDEO_GAME|CONSOLE_VIDEO_GAMES|GAME_|PRELOADED_ELECTRONIC_GAME",
    "MEDIA_BOOKS": r"ABIS_BOOK|ABIS_EBOOKS|ABIS_MUSIC|PHYSICAL_MOVIE|PHYSICAL_TV_SERIES|ABIS_VIDEO|BOOK_",
    "OTT_SUB": r"STREAMING_VIDEO_SUBSCRIPTION|STREAMING_MUSIC_SUBSCRIPTION|KINDLE_UNLIMITED_MEMBERSHIP",
    "GIFT_CARD": r"^GIFT_CARD$",
}


def _share(sub, pattern):
    """Share of spend in `sub` (order rows) matching `pattern`."""
    if len(sub) == 0:
        return 0.0
    mask = sub["category"].str.contains(pattern, case=False, na=False)
    tot = sub["spend"].sum()
    return float(sub.loc[mask, "spend"].sum() / tot) if tot > 0 else 0.0


def _flag(sub, pattern):
    return int(sub["category"].str.contains(pattern, case=False, na=False).any()) if len(sub) else 0


HF_DATA_REPO = "vivekkopthsd/churn-predict-data"  # large raw files are hosted on Hugging Face


def _ensure_raw(filename):
    """Return a local path for `filename`, downloading from the HF dataset if absent."""
    path = os.path.join(RAW, filename)
    if os.path.exists(path):
        return path
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_DATA_REPO, filename=os.path.join("data", "raw", filename))
    except Exception as exc:
        raise FileNotFoundError(
            f"Missing {filename}. Download it from "
            "https://huggingface.co/datasets/vivekkopthsd/churn-predict-data "
            f"into {RAW} (or run `pip install -U huggingface_hub`)."
        ) from exc


def load_raw():
    orders = pd.read_parquet(_ensure_raw("ordered_open_ecommerce.parquet"))
    survey = pd.read_csv(_ensure_raw("survey.csv"))
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders = orders.dropna(subset=["price", "quantity", "order_date"])
    orders["spend"] = orders["price"] * orders["quantity"]
    return orders, survey


def demographics(survey):
    dem = pd.DataFrame({"Customer_ID": survey["Survey ResponseID"].astype(str)})
    dem["Age"] = survey["Q-demos-age"].map(AGE_MID)
    dem["Annual_Income_USD"] = survey["Q-demos-income"].map(INCOME_MID)
    dem["Gender"] = survey["Q-demos-gender"].map(lambda x: "Other" if x in ("Other", "Prefer not to say") else x)
    dem["Education_Level"] = survey["Q-demos-education"].map(EDU_MAP).fillna("Other")
    dem["Household_Size"] = survey["Q-amazon-use-hh-size"].map(HH_MAP)
    dem["Race_Group"] = survey["Q-demos-race"].apply(lambda r: "Multiracial" if (isinstance(r, str) and "," in r)
                                                     else ("White only" if r == "White or Caucasian" else
                                                           ("Black only" if r == "Black or African American" else
                                                            ("Asian only" if r == "Asian" else "Other"))))
    dem["Hispanic"] = (survey["Q-demos-hispanic"] == "Yes").astype(int)
    dem["Reported_Usage_Freq"] = survey["Q-amazon-use-how-oft"].map(FREQ_MAP)
    st = survey["Q-demos-state"].astype(str).str.strip().map(STATE_ABBR)
    dem["Region"] = st.map(REGION).fillna("Unknown")
    return dem


def temporal_snapshot(cutoff, orders, dem):
    """Features measured ONLY from orders before `cutoff`; label = no purchase in next 90 days."""
    before = orders[orders["order_date"] < cutoff].copy()
    win = orders[(orders["order_date"] >= cutoff) & (orders["order_date"] < cutoff + pd.Timedelta(days=OUTCOME_DAYS))]
    churned = set(win["survey_id"].astype(str).unique())

    lo, hi = cutoff - pd.Timedelta(days=365), cutoff
    recent_lo, recent_hi = cutoff - pd.Timedelta(days=90), cutoff
    prior_lo, prior_hi = cutoff - pd.Timedelta(days=180), cutoff - pd.Timedelta(days=90)
    b12 = before[(before["order_date"] >= lo) & (before["order_date"] < hi)]
    rec = before[(before["order_date"] >= recent_lo) & (before["order_date"] < recent_hi)]
    pri = before[(before["order_date"] >= prior_lo) & (before["order_date"] < prior_hi)]

    spend12 = b12.groupby("survey_id")["spend"].sum()
    g = b12.groupby("survey_id")
    idx = g.size().index
    shares = b12.groupby("survey_id").apply(lambda s: pd.Series({
        "entertainment_share": _share(s, CAT["ENTERTAINMENT_DIGITAL"]),
        "gaming_share": _share(s, CAT["GAMING"]),
        "media_share": _share(s, CAT["MEDIA_BOOKS"]),
        "gift_card_share": _share(s, CAT["GIFT_CARD"]),
        "ott_flag": _flag(s, CAT["OTT_SUB"]),
    })).reindex(idx)
    feat = pd.DataFrame({
        "Customer_ID": idx.astype(str),
        "orders_12m": g.size().values,
        "spend_12m": spend12.values,
        "aov_12m": g["spend"].mean().values,
        "aov_recent": rec.groupby("survey_id")["spend"].mean().reindex(idx).values,
        "aov_prior": pri.groupby("survey_id")["spend"].mean().reindex(idx).values,
        "recency_days": (cutoff - g["order_date"].max()).dt.days.values,
        "active_months_12m": g["order_date"].apply(lambda s: s.dt.to_period("M").nunique()).values,
        "category_count_12m": g["category"].nunique().values,
        "avg_price_12m": g["price"].mean().values,
    })
    feat = feat.join(shares.reset_index(drop=True))
    feat = feat.merge(dem, on="Customer_ID", how="inner")
    feat["aov_trend"] = feat["aov_recent"] - feat["aov_prior"]
    # Churn labels at multiple outcome windows (fully observable before data end 2023-03-19)
    for w in (30, 60, 90, 120, 180):
        win_w = orders[(orders["order_date"] >= cutoff) & (orders["order_date"] < cutoff + pd.Timedelta(days=w))]
        churned_w = set(win_w["survey_id"].astype(str).unique())
        feat[f"label_churn_{w}"] = (~feat["Customer_ID"].isin(churned_w)).astype(int)
    feat["label_churn"] = feat["label_churn_90"]
    feat = feat.dropna(subset=["orders_12m"])
    feat["orders_12m"] = feat["orders_12m"].astype(int)
    feat["active_months_12m"] = feat["active_months_12m"].astype(int)
    return feat.reset_index(drop=True)


def survival_labels(cutoff, orders):
    """Time to FIRST PURCHASE after cutoff (time-to-return). The survival event is a RETURN,
    NOT a churn: event=1 if the customer returned before data end, event=0 (censored, i.e.
    did not return) otherwise, with T set to the full observation horizon for censored rows.
    These labels are used for time-to-first-purchase survival models; S(t) = P(no return by t)."""
    aft = orders[orders["order_date"] >= cutoff]
    first = aft.groupby("survey_id")["order_date"].min()
    out = pd.DataFrame({"Customer_ID": first.index.astype(str)})
    out["T_days"] = (first.values - np.datetime64(cutoff)).astype("timedelta64[D]").astype(int)
    out["event"] = 1
    all_cust = set(orders["survey_id"].astype(str))
    cens = pd.DataFrame({"Customer_ID": [c for c in all_cust if c not in set(out["Customer_ID"])]})
    cens["T_days"] = (DATA_END - cutoff).days
    cens["event"] = 0
    return pd.concat([out, cens], ignore_index=True)


def full_customer_table(orders, dem):
    """Full-history customer table: demographics + behavioural + category + CLV-at-risk value."""
    g = orders.groupby("survey_id")
    first, last = g["order_date"].min(), g["order_date"].max()
    beh = pd.DataFrame({
        "Customer_ID": g.size().index.astype(str),
        "Total_Spend_USD": g["spend"].sum().values,
        "Order_Count": g.size().values,
        "Avg_Order_Value_USD": g["spend"].mean().values,
        "Active_Months": g["order_date"].apply(lambda s: s.dt.to_period("M").nunique()).values,
        "Category_Count": g["category"].nunique().values,
        "Recency_Days": (pd.Timestamp("2023-03-19") - last).dt.days.clip(lower=0).values,
    })
    beh["Span_Years"] = (((last - first).dt.days + 1) / 365.25).values
    beh["Orders_Per_Year"] = beh["Order_Count"] / beh["Span_Years"].clip(lower=1 / 365.25)
    beh["CLV_3Yr"] = (beh["Total_Spend_USD"] / beh["Span_Years"].clip(lower=1 / 365.25) * 3).round(2)
    # Winsorize at p99: a few short-span heavy spenders create $100k+ velocity-CLV outliers
    # that would let ONE customer dominate the offer-budget optimization.
    _cap = float(beh["CLV_3Yr"].quantile(0.99))
    beh["CLV_3Yr"] = beh["CLV_3Yr"].clip(upper=_cap)
    print(f"[integrity] CLV winsorized at p99 = ${_cap:,.0f}")
    idx = g.size().index
    shares = orders.groupby("survey_id").apply(lambda s: pd.Series({
        "entertainment_share": _share(s, CAT["ENTERTAINMENT_DIGITAL"]),
        "ott_flag": _flag(s, CAT["OTT_SUB"]),
    })).reindex(idx)
    beh = beh.join(shares.reset_index(drop=True))
    df = dem.merge(beh, on="Customer_ID", how="inner")
    df = df[df["Order_Count"] >= 3]
    return df.reset_index(drop=True)


def main():
    orders, survey = load_raw()
    dem = demographics(survey)

    print(f"Orders: {len(orders):,} | survey respondents: {len(dem):,}")

    tr = temporal_snapshot(CUTOFF_TRAIN, orders, dem)
    te = temporal_snapshot(CUTOFF_TEST, orders, dem)
    print(f"\nTrain snapshot (cutoff {CUTOFF_TRAIN.date()}): {len(tr):,} customers, "
          f"churn base rate {tr.label_churn.mean():.1%}")
    print(f"Test  snapshot (cutoff {CUTOFF_TEST.date()}): {len(te):,} customers, "
          f"churn base rate {te.label_churn.mean():.1%}")
    print(f"Overlap of customers between snapshots: {len(set(tr.Customer_ID) & set(te.Customer_ID)):,} "
          f"(shared customers are fine - features/labels are period-specific)")

    cust = full_customer_table(orders, dem)
    print(f"\nCustomer table (full history): {len(cust):,} x {cust.shape[1]}")

    s_tr = survival_labels(CUTOFF_TRAIN, orders)
    s_te = survival_labels(CUTOFF_TEST, orders)
    print(f"Survival labels - train events {s_tr.event.sum():,} / {len(s_tr):,} | "
          f"test events {s_te.event.sum():,} / {len(s_te):,} (censored = did not return by data end)")
    s_tr.to_csv(os.path.join(OUT, "survival_train.csv"), index=False)
    s_te.to_csv(os.path.join(OUT, "survival_test.csv"), index=False)
    tr.to_csv(os.path.join(OUT, "snapshot_train.csv"), index=False)
    te.to_csv(os.path.join(OUT, "snapshot_test.csv"), index=False)
    cust.to_csv(os.path.join(OUT, "customers.csv"), index=False)
    print("\nSaved: data/snapshot_train.csv, data/snapshot_test.csv, data/customers.csv, "
          "data/survival_train.csv, data/survival_test.csv")


if __name__ == "__main__":
    main()
