# CTR scoring notebook 4: does the flag persist into the next month? (warehouse, two windows)
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
r = RAW[MONTHS[0]]
print("\nPosition scale check (median of pos, median of sum_position/impressions, median difference):",
      round(float(r["pos"].median()), 2), round(float(r["pos_from_sum"].median()), 2),
      round(float((r["pos"] - r["pos_from_sum"]).median()), 2))

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
    d.attrs["phi"] = float(np.median(d["pearson"] ** 2) / 0.4549)
    d["p_low"] = norm.cdf(d["pearson"] / np.sqrt(max(d.attrs["phi"], 1.0)))
    d["missed"] = (d["exp_clicks"] - d["clicks"]).clip(lower=0)
    d["flag"] = (d["p_low"] < 0.01) & (d["missed"] >= 5)
    d["ratio"] = (d["clicks"] + 0.5) / (d["exp_clicks"] + 0.5)
    return d

SC = {m: score(RAW[m]) for m in MONTHS}
for m in MONTHS:
    print(f"{m}: kept {len(SC[m]):,} pages | phi {SC[m].attrs['phi']:.1f} | flagged {SC[m]['flag'].sum():,} ({SC[m]['flag'].mean():.1%})")

OUT = []
for a, b in PAIRS:
    A, B = SC[a], SC[b]
    M = A.merge(B, on="content_hash_id", suffixes=("_a", "_b"))
    fa = M["flag_a"]
    extreme = (M["clicks_a"] < 0.05 * M["exp_clicks_a"]) & (M["exp_clicks_a"] >= 1000)
    fa_clean = fa & ~extreme
    rest = ~fa
    line = {}
    line["pair"] = f"{a} -> {b}"
    line["pages_in_both"] = len(M)
    line["flagged_in_A"] = int(fa.sum())
    line["flagged_in_A_not_extreme"] = int(fa_clean.sum())
    line["median_ratio_B_flagged"] = round(float(M.loc[fa, "ratio_b"].median()), 3)
    line["median_ratio_B_flagged_not_extreme"] = round(float(M.loc[fa_clean, "ratio_b"].median()), 3)
    line["median_ratio_B_rest"] = round(float(M.loc[rest, "ratio_b"].median()), 3)
    line["share_B_under_half_expected_flagged"] = round(float((M.loc[fa, "ratio_b"] < 0.5).mean()), 3)
    line["share_B_under_half_expected_rest"] = round(float((M.loc[rest, "ratio_b"] < 0.5).mean()), 3)
    line["flagged_again_in_B_flagged"] = round(float(M.loc[fa, "flag_b"].mean()), 3)
    line["flagged_in_B_rest"] = round(float(M.loc[rest, "flag_b"].mean()), 3)
    line["spearman_log_ratio_A_vs_B"] = round(float(spearmanr(np.log(M["ratio_a"]), np.log(M["ratio_b"]))[0]), 3)
    line["share_of_pages_flagged_A"] = round(float(fa.mean()), 3)
    line["share_of_B_missed_clicks_in_flagged_A"] = round(float(M.loc[fa, "missed_b"].sum() / M["missed_b"].sum()), 3)
    line["clients_with_flagged_A"] = int(M.loc[fa, "client_hash_id_a"].nunique())
    OUT.append(line)

print("\n===== COPY EVERYTHING BELOW THIS LINE AND SEND IT TO ME =====")
for line in OUT:
    print()
    for k, v in line.items():
        print(f"{k}: {v}")
#### END PART 2
