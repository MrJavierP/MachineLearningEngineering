import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")                      # no display inside the container
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


MONITORING_DIR = "datamart/gold/model_monitoring/"
PLOTS_DIR = MONITORING_DIR + "plots/"

PSI_WATCH, PSI_RETRAIN = 0.10, 0.25        # same thresholds as model_monitor.py
AUC_FLOOR = 0.85


def load_monitoring():
    # one row per model per month, across every model folder (champion included)
    frames = []
    for path in sorted(glob.glob(MONITORING_DIR + "*/*.parquet")):
        model_id = os.path.basename(os.path.dirname(path))
        df = pd.read_parquet(path)
        df["model_id"] = model_id
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df.sort_values(["model_id", "snapshot_date"])


def _finish(fig, ax, out):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_performance(df, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_id, g in df.groupby("model_id"):
        ax.plot(g["snapshot_date"], g["auc"], marker="o", label=model_id)
        if "auc_oot_at_training" in g and g["auc_oot_at_training"].notna().any():
            ax.axhline(g["auc_oot_at_training"].iloc[0], ls=":", lw=1,
                       color=ax.lines[-1].get_color())
    ax.axhline(AUC_FLOOR, ls="--", lw=1, color="red", label="retrain floor (0.85, 2 months)")
    ax.set_title("Model performance on matured cohorts (AUC by month)")
    ax.set_ylabel("AUC"); ax.set_xlabel("label snapshot month")
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _finish(fig, ax, out)


def plot_stability(df, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_id, g in df.groupby("model_id"):
        ax.plot(g["snapshot_date"], g["psi_vs_deployment"], marker="o", label=model_id)
    ax.axhline(PSI_WATCH, ls="--", color="orange", lw=1, label="watch (0.10)")
    ax.axhline(PSI_RETRAIN, ls="--", color="red", lw=1, label="retrain (0.25)")
    ax.set_title("Score stability vs deployment month (PSI)")
    ax.set_ylabel("PSI"); ax.set_xlabel("scoring month")
    ax.set_ylim(0, max(0.3, df["psi_vs_deployment"].max() * 1.2)); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _finish(fig, ax, out)


def plot_calibration(df, out):
    # does the average predicted default rate track the default rate that actually arrived?
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_id, g in df.groupby("model_id"):
        ax.plot(g["snapshot_date"], g["avg_predicted_prob"], marker="o", label=f"{model_id} predicted")
    g0 = df.drop_duplicates("snapshot_date")
    ax.plot(g0["snapshot_date"], g0["actual_default_rate"], marker="s", color="black", lw=2, label="actual default rate")
    ax.set_title("Predicted vs actual default rate on matured cohorts")
    ax.set_ylabel("default rate"); ax.set_xlabel("label snapshot month")
    ax.set_ylim(0, 0.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _finish(fig, ax, out)


def main(snapshotdate):
    print('\n\n---starting model visualisation job---\n\n')
    df = load_monitoring()
    if df.empty:
        print("skip: no monitoring rows yet")
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_performance(df, PLOTS_DIR + "performance_auc.png")
    plot_stability(df, PLOTS_DIR + "stability_psi.png")
    plot_calibration(df, PLOTS_DIR + "calibration.png")

    # a flat summary table for the deck and for anyone without a parquet reader
    cols = [c for c in ["snapshot_date", "model_id", "cohort_application_month", "n_labels_matured",
                        "auc", "gini", "actual_default_rate", "avg_predicted_prob",
                        "n_scored_this_month", "psi_vs_deployment", "auc_drop", "status"] if c in df]
    summary = df[cols].copy()
    summary["snapshot_date"] = summary["snapshot_date"].dt.date
    summary.to_csv(PLOTS_DIR + "monitoring_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("plots and summary written to", PLOTS_DIR, "as of", snapshotdate)
    print('\n\n---model visualisation job completed---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot model performance and stability across months")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
