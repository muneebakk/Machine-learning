# CTR scoring notebook 6: does the A-ranking beat a size-only ranking at covering next month's shortfall?
# Paste this whole file into ONE Colab cell and press Run. Needs the Colab secret HF_TOKEN (Notebook access ON).
# It reads only three monthly files, aggregates them per page in DuckDB, and prints NO ids and NO client info.

#### PART 1: load (needs internet + token)
!pip install -q duckdb huggingface_hub
import duckdb
import numpy as np
import pandas as pd
from google.colab import userdata
from scipy.stats import norm, poisson, spearmanr

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute(f"CREATE SECRET hf_secret (TYPE HUGGINGFACE, TOKEN '{userdata.get('HF_TOKEN')}')")
BASE = "hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance"

PAIRS = [("2026-03", "2026-04"), ("2026-04", "2026-05")]   # (window A = used to flag, window B = next month, used to check)
MONTHS = sorted({m for p in PAIRS for m in p})

def load_month(m):
    q = f'''
    SELECT client_hash_id, content_hash_id,
           sum(gsc_impressions) AS impressions, sum(gsc_clicks) AS clicks,
           sum(gsc_avg_position * gsc_impressions) / sum(gsc_impressions) AS pos,
           sum(gsc_sum_position) * 1.0 / sum(gsc_impressions) AS pos_from_sum
    FROM read_parquet('{BASE}/month={m}/data_0.parquet')
    WHERE gsc_impressions > 0
    GROUP BY 1, 2
    '''
    return con.sql(q).df()

RAW = {}
for m in MONTHS:
    RAW[m] = load_month(m)
    print(m, "pages with impressions:", f"{len(RAW[m]):,}", "| clients:", RAW[m]["client_hash_id"].nunique())
#### END PART 1

#### PART 2: analysis (uses RAW only)
BINS = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 10.5, 15.5, 20.5, 1000]
K = 50_000
MIN_IMP = 100

def score(raw):
    d = raw[(raw["impressions"] >= MIN_IMP) & raw["pos"].notna()].copy()
    d["ctr"] = d["clicks"] / d["impressions"]
    d["bucket"] = pd.cut(d["pos"], bins=BINS, labels=False)
    g = d.groupby("bucket")[["clicks", "impressions"]].sum()
    d["global_ctr"] = d["bucket"].map(g["clicks"] / g["impressions"])
    cb = d.groupby(["client_hash_id", "bucket"])[["clicks", "impressions"]].transform("sum")
    li, lc = cb["impressions"] - d["impressions"], cb["clicks"] - d["clicks"]
    loo = np.where(li > 0, lc / li.where(li > 0, 1), 0.0)
    w = li / (li + K)
    d["exp_ctr"] = np.clip(w * loo + (1 - w) * d["global_ctr"], 1e-6, 1)
    d["exp_clicks"] = d["exp_ctr"] * d["impressions"]
    d["pearson"] = (d["clicks"] - d["exp_clicks"]) / np.sqrt(d["exp_clicks"])
    phi = float(np.median(d["pearson"] ** 2) / 0.4549)
    d["p_low"] = norm.cdf(d["pearson"] / np.sqrt(max(phi, 1.0)))
    d["missed"] = (d["exp_clicks"] - d["clicks"]).clip(lower=0)
    d["flag"] = (d["p_low"] < 0.01) & (d["missed"] >= 5)
    d["ratio"] = (d["clicks"] + 0.5) / (d["exp_clicks"] + 0.5)
    return d


SC = {m: score(RAW[m]) for m in MONTHS}
OUT = []
for a, b in PAIRS:
    M = SC[a].merge(SC[b], on="content_hash_id", suffixes=("_a", "_b"))
    tot = M["missed_b"].sum()
    n = len(M)
    ranks = {
        "A: biggest estimated missed clicks (our ranking)": M.sort_values("missed_a", ascending=False),
        "size only: most impressions": M.sort_values("impressions_a", ascending=False),
        "gap only: lowest clicks/expected ratio": M.sort_values("ratio_a", ascending=True),
    }
    line = {"pair": f"{a} -> {b}"}
    for name, R in ranks.items():
        line[name] = {f"top {int(q*100)}%": round(float(R["missed_b"].head(int(n * q)).sum() / tot), 3) for q in [0.01, 0.02, 0.05, 0.10]}
    OUT.append(line)

print("===== COPY EVERYTHING BELOW THIS LINE AND SEND IT TO ME =====")
for line in OUT:
    print()
    for k, v in line.items():
        print(f"{k}: {v}")
#### END PART 2
