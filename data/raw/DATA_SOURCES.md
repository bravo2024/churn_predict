# Data Sources — Open E-Commerce 1.0 (real data)

| File | Source | License | Verified |
|:--|:--|:--|:--|
| `ordered_open_ecommerce.parquet` (158 MB, 1,761,259 rows) | HuggingFace dataset `mpa326/open_ecommerce` → `https://huggingface.co/datasets/mpa326/open_ecommerce/resolve/main/ordered_open_ecommerce.parquet` | MIT (ungated) | 5,025 unique customers; 2018-01-01 → 2024-08-15 (data collection effectively ends 2023-03-19; single 2024 outlier); no null price/quantity/date; no non-positive prices |
| `survey.csv` (1.3 MB, 5,027 rows) | GitHub `kongzikun/Amazon-Consumer-Behavior-Analysis` → `https://raw.githubusercontent.com/kongzikun/Amazon-Consumer-Behavior-Analysis/main/data/survey.csv` | MIT (repo license) | 5,027 unique response IDs; 5,025/5,025 purchasers matched (99.96% join) |

**Citation:** Berke et al. (2024), MIT Media Lab — "Open e-commerce 1.0" data release
(1.8M Amazon purchase records + demographic survey of U.S. shoppers).

**Build:** `python build_dataset.py` produces `data/customers.csv` (5,013 customers × 24+
columns) — demographics parsed from survey buckets, behavioral aggregates from purchases,
targets defined as documented proxies (velocity CLV, activity-composite engagement,
recency-based churn).

**Hosting:** the 158 MB parquet exceeds GitHub's 100 MB per-file limit, so the large raw
and model files are mirrored on Hugging Face at
`vivekkopthsd/churn-predict-data` (https://huggingface.co/datasets/vivekkopthsd/churn-predict-data).
`build_dataset.py` auto-downloads them when missing locally.

**Ethics:** only benign demographics + Amazon-usage survey items are used; sensitive
survey items (health, substance use, data-selling attitudes) are excluded from analysis.
