"""Contract tests for the shipped artifact and the dashboard's computed figures.

Run directly (no pytest needed):

    python tests/test_published_figures.py

The dashboard reads every model figure from models/artifacts_slim.pkl. These tests pin
the numbers the published documents quote, so a regenerated artifact cannot silently
change what the dashboard claims, and they fail if a retired hardcoded figure creeps
back into dashboard.py.
"""
import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SLIM = os.path.join(BASE, "models", "artifacts_slim.pkl")
DASH = os.path.join(BASE, "dashboard.py")


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def load():
    with open(SLIM, "rb") as fh:
        return pickle.load(fh)


def main():
    a = load()
    cs = a["churn_summary"]
    bench = a["churn_metrics_df"]
    offer = a["offer_result"]
    sens = a["sensitivity_df"]

    # Published generation-B figures.
    assert cs["test_customers"] == 4850, cs["test_customers"]
    assert cs["actual_churned"] == 481, cs["actual_churned"]
    assert cs["top10_list_size"] == 485, cs["top10_list_size"]
    assert close(cs["base_rate"], 0.0992, 0.0005), cs["base_rate"]
    assert close(bench.loc["Voting (Ch7)", "Prec@10%"], 0.487, 0.001)
    assert close(bench.loc["Voting (Ch7)", "ROC-AUC"], 0.898, 0.001)
    assert close(offer["opt_rev"], 202403, 1), offer["opt_rev"]
    assert close(offer["roas"], 20.24, 0.01), offer["roas"]
    assert close(offer["greedy_rev"], 80543, 1), offer["greedy_rev"]
    assert close(offer["rand_rev"], 8385, 1), offer["rand_rev"]
    assert offer["budget"] == 10000, offer["budget"]
    assert int(offer["customers_offered"]) == 400, offer["customers_offered"]
    auc = sens["AUC (temporal test)"]
    assert 0.86 <= float(auc.min()) <= float(auc.max()) <= 0.90, auc.tolist()

    # The figures the CV quotes (cv_assurant.tex): 1.76M records, 5,013 customers,
    # 24 features, voting ensemble, ROC-AUC 0.898, 0.487 top-decile precision, 0.87-0.89.
    assert len(a["cust"]) == 5013, len(a["cust"])
    assert round(float(bench.loc["Voting (Ch7)", "ROC-AUC"]), 3) == 0.898
    assert round(float(bench.loc["Voting (Ch7)", "Prec@10%"]), 3) == 0.487
    assert (round(float(auc.min()), 2), round(float(auc.max()), 2)) == (0.87, 0.89)
    assert len(a["sensitivity_df"]) == 5, "CV says five churn-window definitions"

    # The list the dashboard calls out must be the list the artifact scored.
    assert cs["top10_list_size"] == 485 and abs(cs["top10_precision"] - 0.4804) < 0.001

    # Retired hardcoded prose that must not return.
    src = open(DASH, encoding="utf-8").read()
    retired = [
        r"5x precision", r"median of 221", r"9\.9 percent", r"202,403", r"80,543",
        r"0\.87-0\.89", r"1,455", r"About 49%", r"about 60% below", r"\bamz-",
        r"High-Value Loyalists", r"min\(0\.60,\\?; 0\.04", r"def response\(d, base=",
    ]
    hits = [p for p in retired if re.search(p, src)]
    assert not hits, f"retired hardcoded figures in dashboard.py: {hits}"

    for needed in ["response(", "OFFERS", "RESPONSE", "min_detectable_effect",
                   "voting_test_eval", "list_precision", "panel("]:
        assert needed in src, f"dashboard.py lost {needed}"

    print("OK  artifact and dashboard contract hold")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL  {exc}")
        sys.exit(1)
