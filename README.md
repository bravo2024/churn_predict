# Customer Churn Prediction and Retention-Budget Optimization

Customer churn prediction and retention-budget optimization on real e-commerce data. The
project predicts which customers lapse within a 90-day window, explains the drivers of
churn, models time-to-return with survival analysis, and allocates a limited retention
budget with an exact Mixed-Integer Linear Program (MILP).

**Data:** Open E-Commerce 1.0 (Berke et al., 2024, *Scientific Data*,
DOI 10.1038/s41597-024-03329-6) - 1,761,259 raw purchase records aggregated to a
5,013-customer analysis table.

## Key results

| Result | Value |
|:--|--:|
| Churn model ROC-AUC (temporal test period) | 0.898 |
| Precision at top-10% targeting decile | ~0.49 (5x the 9.9% base rate) |
| AUC stability across 30-180-day churn windows | 0.87-0.89 |
| Retention budget allocation (MILP, $10,000 budget) | ~$202k retained value, projected ROAS ~20x |

The ROAS figure is a projection under an assumed offer-response curve
`clip(0.04 + 0.018*d, 0, 0.60)`, not a measured return. The A/B test design in this
repository specifies how that curve would be validated with observed response data.

## Data

The raw source is the Open E-Commerce 1.0 dataset (Berke et al., 2024): crowdsourced
Amazon purchase histories and a demographic survey of U.S. shoppers, both MIT-licensed.
1,761,259 purchase records (2018-01 to 2023-03) are aggregated into a customer-level
table of 5,013 customers with 24 behavioral and demographic features, split temporally
into 4,857 train and 4,850 test customers. The 90-day churn base rate is 9.9 percent.

The two large files (`data/raw/ordered_open_ecommerce.parquet`, ~150 MB, and
`models/artifacts.pkl`, ~200 MB) exceed GitHub's 100 MB per-file limit and are hosted on
Hugging Face at [vivekkopthsd/churn-predict-data](https://huggingface.co/datasets/vivekkopthsd/churn-predict-data).
They are downloaded automatically on first use. See `data/raw/DATA_SOURCES.md` for full
provenance.

## Methodology

1. **Temporal validation.** The train period ends 2022-03-19 and the test period begins
   2022-09-19. The windows are disjoint in time, every outcome window is fully observed,
   and no feature uses post-cutoff information. The two snapshots score the same customer
   base at two temporal cutoffs; this is not a customer-holdout evaluation, and features
   are strictly pre-cutoff, so no future information leaks into either snapshot.
2. **Churn definition.** Churn is defined as no purchase within N days, with N = 90 as the
   primary window and N in {30, 60, 120, 180} used for sensitivity analysis.
3. **Leakage control.** Features are computed from orders before each cutoff. CLV is
   winsorized at the 99th percentile and multicollinearity is checked with VIF.
4. **Business metric.** Precision at the top-10% score decile is the primary targeting
   metric, because churn is rare (~10 percent) and accuracy alone is not informative.
5. **Ethics.** Only benign demographic and Amazon-usage survey items are used. Sensitive
   health and substance-use items are excluded.

## Analysis modules

### 1. Churn prediction

Models are trained on the train period and evaluated on the unseen test period.

| Model | ROC-AUC | Precision at top-10% |
|:--|--:|--:|
| Voting (Logistic + RF + GB) | **0.898** | 0.487 |
| XGBoost | 0.893 | 0.480 |
| Random Forest | 0.895 | 0.497 |
| Logistic | 0.895 | 0.485 |
| Gradient Boosting | 0.891 | 0.460 |
| Decision Tree | 0.873 | 0.499 |
| Baseline (no model) | 0.500 | 0.099 |

The top-decile targeting list of 485 customers contains 233 real churners, a 48 percent
precision that is about five times the base rate. AUC is stable between 0.87 and 0.89
across 30 to 180-day churn windows, and the drivers are confirmed by split importance,
permutation importance, and SHAP values.

**Drivers of churn.** Protective factors include active months (r ~ -0.50), order velocity,
spend, and category breadth (r ~ -0.29); OTT/streaming subscribers churn at 3.3 percent
versus 10.9 percent overall, indicating a recurring-intent segment. Leading indicators
include recency at cutoff (r ~ +0.49) and a declining average-order-value trend.
Demographics matter less than behavior.

**Customer segments** (K-Means on 6 principal components):

| Segment | n | Mean CLV | Churn risk | Recommended action |
|:--|--:|--:|:--|:--|
| High-Value Loyalists | 528 | $16,250 | low | Protect; premium offers only |
| Established Shoppers | 153 | $7,384 | low | OTT/subscription buyers; cross-sell |
| Occasional Shoppers | 2,088 | $5,996 | medium | Re-engagement and category cross-sell |
| Light / Disengaged | 2,244 | $1,867 | high | Nurture; low-cost win-back offers |

### 2. Survival analysis (time-to-return)

The event modeled is time to first purchase after a cutoff (time-to-return), with
censoring explicit: customers who never returned before the data end are censored. A
Random Survival Forest and a Cox baseline were fitted. S(t) denotes the probability of no
return by day t: S(30) ~ 0.25, S(90) ~ 0.11, and S(180) ~ 0.05 on the test period, and
S(90) correlates at 0.94 with the binary churn probability.

These results are indicative rather than a clean out-of-sample validation: the same
customer cohort appears in the train and test snapshots, and the outcome windows overlap
in calendar time, so the concordance figures (RSF 0.727, Cox 0.744) are treated as
in-sample only.

### 3. Retention offer optimization (MILP)

The retention budget is allocated exactly as a 0/1-knapsack Mixed-Integer Linear Program
solved with HiGHS. Expected retained value is the churn probability times the CLV at risk
times the response to an offer, with the response curve stated as an assumption to be
validated by experiment. At a $10,000 budget the exact allocation retains about $202,403
of value (projected ROAS 20.24 under the assumed response curve), versus $80,539 for the
greedy heuristic (about 60.2 percent below the optimal solution) and $67,516 for naive
spend.

*Note on numbers.* These values come from the executed notebook, which scores the MILP
with the voting-ensemble probabilities. The allocation is a solver run, so exact values
vary run to run; the current `models/artifacts.pkl` was trained from Gradient-Boosting
scores and holds a different allocation at the same budget (about $220,409 retained, ROAS
22.0) - the dashboard labels this provenance under the optimizer tab.

### 4. A/B test design

The win-back experiment is designed with a two-proportion z-test. Sample-size and power
formulas are derived, and the test-statistic and confidence-interval formulas for the
observed difference are stated. The worked example (z = 2.93, p = 0.0034, 95 percent
interval of +3.5 to +17.5 percentage points) is an illustrative power example with
synthetic numbers, not an executed experiment. At 12 percentage points the 485-customer
top-decile list (242 per group) reaches 77 percent power (93 percent at 15 pp); 80 percent
power is crossed just above 12 pp, so the recommended rollout is the top three deciles,
about 1,455 customers.

## Reproducibility

```bash
pip install -r requirements.txt   # Python 3.11

python build_dataset.py      # raw data -> customer table + temporal snapshots + survival labels
python build_notebook.py     # rebuild the notebook
python execute_notebook.py   # execute in place
python train_models.py       # models and artifacts for the dashboard
streamlit run dashboard.py   # interactive dashboard (scope-selectable)
```

All random seeds are 42. Environment: Python 3.11, pandas 2.2, scikit-learn 1.9,
xgboost 3.2, scikit-survival 0.28, shap 0.51, streamlit 1.53.

## Repository structure

```
├── build_dataset.py          # data engineering, temporal labels, survival labels, winsorization
├── build_notebook.py         # notebook generator
├── execute_notebook.py       # papermill executor
├── train_models.py           # models, SHAP, survival, MILP optimizer, artifacts
├── dashboard.py              # Streamlit dashboard with analysis-scope selector
├── amazon_churn_retention_analysis.ipynb   # executed analysis notebook (outputs embedded)
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/                  # survey.csv (see DATA_SOURCES.md); parquet auto-downloaded from HF
│   ├── snapshot_train.csv, snapshot_test.csv, customers.csv
│   └── survival_train.csv, survival_test.csv
└── models/
    ├── artifacts.pkl         # fitted models, SHAP, survival, MILP results (auto-downloaded from HF)
    ├── churn_scores_test.csv # per-customer churn scores (test period)
    └── offer_allocation.csv  # priority win-back list with offer sizes
```

## Data availability

The large raw data and fitted artifacts are hosted on Hugging Face:

- Dataset: [vivekkopthsd/churn-predict-data](https://huggingface.co/datasets/vivekkopthsd/churn-predict-data)
- `data/raw/ordered_open_ecommerce.parquet` (150 MB) - raw purchase records
- `models/artifacts.pkl` (~200 MB) - fitted models, SHAP, survival, and MILP results

`build_dataset.py` and `dashboard.py` download them automatically on first use; the
dashboard falls back to retraining the suite if the download is unavailable.

## License

MIT. See [LICENSE](LICENSE). The underlying data is MIT-licensed (Open E-Commerce 1.0,
Berke et al., 2024).

## References

1. Berke, A., Calacci, D., Mahari, R., Yabe, T., Larson, K. and Pentland, S. (2024).
   Open e-commerce 1.0, five years of crowdsourced U.S. Amazon purchase histories with
   user demographics. Scientific Data, 11. DOI 10.1038/s41597-024-03329-6.
2. Lundberg, S. and Lee, S.-I. (2017). A unified approach to interpreting model
   predictions. NeurIPS. (SHAP)
