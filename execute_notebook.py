"""Execute the churn-retention notebook in-place (papermill, embeds outputs)."""
import os
import papermill as pm

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "amazon_churn_retention_analysis.ipynb")


def main():
    print(f"Executing {NB} ...")
    pm.execute_notebook(NB, NB, kernel_name="python3", execution_timeout=1800, progress_bar=True)
    print("Done.")


if __name__ == "__main__":
    main()
