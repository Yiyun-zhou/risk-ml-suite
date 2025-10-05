from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv", parse_dates=["event_time"])
    outdir = ROOT / "reports/figures"; outdir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    train["Class"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (train)"); plt.xlabel("Class"); plt.ylabel("Count")
    fig.savefig(outdir / "class_balance.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure()
    train["Amount"].clip(upper=train["Amount"].quantile(0.99)).hist(bins=50)
    plt.title("Amount (train, capped 99%)"); plt.xlabel("Amount")
    fig.savefig(outdir / "amount_hist.png", dpi=160, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
