# src/data_prep.py
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def ensure_kaggle_csv(raw_dir: Path, download: bool = True) -> Path:
    """
    返回 data/raw/creditcard.csv。若不存在且允许下载，则使用 Kaggle CLI:
      kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --force
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "creditcard.csv"
    if csv_path.exists():
        return csv_path
    if not download:
        raise FileNotFoundError(f"{csv_path} 不存在，且未启用自动下载（--download 0）")

    # 尝试下载
    print("[INFO] Downloading Kaggle dataset mlg-ulb/creditcardfraud ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", str(raw_dir), "--force"],
        check=True,
    )
    # 解压
    zips = list(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Kaggle 下载未找到 zip 包")
    import zipfile
    with zipfile.ZipFile(zips[0], "r") as z:
        z.extractall(raw_dir)
    for zf in zips:
        zf.unlink(missing_ok=True)

    if not csv_path.exists():
        # 有些镜像文件名大小写不同，兜底查找
        cand = list(raw_dir.glob("*creditcard*.csv"))
        if not cand:
            raise FileNotFoundError("解压后未找到 creditcard.csv")
        cand[0].rename(csv_path)
    return csv_path

def add_event_time(df: pd.DataFrame, time_col: str = "Time", epoch: str = "2013-01-01") -> pd.Series:
    """
    Kaggle 的 Time 是相对首笔交易的秒数。构造一个可读 event_time。
    """
    start = pd.Timestamp(epoch)
    et = start + pd.to_timedelta(df[time_col].astype(float), unit="s")
    return et

def time_order_split(df: pd.DataFrame, tcol: str, valid_frac=0.15, test_frac=0.15):
    """
    按时间顺序切分（不穿插），默认 70%/15%/15%。
    Kaggle 数据约两天窗口，用比例切分更稳妥。
    """
    df = df.sort_values(tcol).reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_frac))
    n_valid = int(round(n * valid_frac))
    n_train = n - n_valid - n_test
    train = df.iloc[:n_train]
    valid = df.iloc[n_train:n_train + n_valid]
    test  = df.iloc[n_train + n_valid:]
    return train, valid, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=int, default=1, help="1=如需则自动用 Kaggle CLI 下载，0=不下载")
    ap.add_argument("--valid_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--epoch", type=str, default="2013-01-01", help="event_time 起点日期（可任意）")
    args = ap.parse_args()

    raw_dir = ROOT / "data/raw"
    proc_dir = ROOT / "data/processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_kaggle_csv(raw_dir, download=bool(args.download))
    df = pd.read_csv(csv_path)

    # 基本列检查
    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"原始数据缺少必要列: {missing}")

    # 补充 event_time & tx_id（不可泄露）
    df["event_time"] = add_event_time(df, time_col="Time", epoch=args.epoch)
    df["tx_id"] = np.arange(len(df))

    # 仅保留建模必需列（V1..V28, Amount, event_time, Class, tx_id）
    # 这样可避免不小心引入未来信息或 ID 泄露字段（如换其他数据源）
    v_cols = [c for c in df.columns if c.startswith("V")]
    keep = ["tx_id", "event_time", "Amount", "Class"] + v_cols
    df = df[keep].copy()

    # 数据字典
    meta = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "target": "Class (1=fraud, 0=legit)",
        "target_distribution": df["Class"].value_counts(normalize=True).to_dict(),
        "time_min": str(df["event_time"].min()),
        "time_max": str(df["event_time"].max()),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "note": "Kaggle Credit Card Fraud; V1..V28 为 PCA 匿名特征；Time -> event_time（合成日历时间）",
    }
    (ROOT / "reports" / "data_dictionary.json").write_text(json.dumps(meta, indent=2))

    # 时间顺序切分（避免未来信息泄露）
    train, valid, test = time_order_split(df, tcol="event_time", valid_frac=args.valid_frac, test_frac=args.test_frac)

    # 落盘
    train.to_csv(proc_dir / "train.csv", index=False)
    valid.to_csv(proc_dir / "valid.csv", index=False)
    test.to_csv(proc_dir / "test.csv", index=False)

    # 控制台摘要
    def dist(x): return x.value_counts(normalize=True).to_dict()
    print("[SPLIT] shapes:", train.shape, valid.shape, test.shape)
    print("[SPLIT] target rate:", dist(train["Class"]), dist(valid["Class"]), dist(test["Class"]))
    print(f"[OK] Saved to {proc_dir}/(train|valid|test).csv")

if __name__ == "__main__":
    main()
