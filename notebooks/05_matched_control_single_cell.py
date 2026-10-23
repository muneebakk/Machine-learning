# CTR scoring notebook 5: matched-control persistence test, concentration, and aggregate numbers for the paper
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
rng = np.random.default_rng(42)

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

def boot_ci(sums, counts, n=2000):
    """cluster bootstrap over clients: sums/counts are per-client arrays"""
    k = len(sums); est = []
    for _ in range(n):
        i = rng.integers(0, k, k)
        est.append(sums[i].sum() / counts[i].sum())
    return np.percentile(est, [2.5, 97.5])

OUT = []
for a, b in PAIRS:
    A, B = SC[a], SC[b]
    M = A.merge(B, on="content_hash_id", suffixes=("_a", "_b"))
    M["client"] = M["client_hash_id_a"]
    fa = M["flag_a"]
    M["lr_b"] = np.log(M["ratio_b"])
    M["fb"] = M["flag_b"].astype(float)
    M["imp_band"] = M.groupby("bucket_a")["impressions_a"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    key = ["client", "bucket_a", "imp_band"]
    nf = (~fa).astype(float)
    M["_n"] = nf; M["_lr"] = M["lr_b"] * nf; M["_fb"] = M["fb"] * nf
    S = M.groupby(key)[["_n", "_lr", "_fb"]].transform("sum")
    ok = fa & (S["_n"] >= 1)
    M["diff_lr"] = np.where(ok, M["lr_b"] - S["_lr"] / S["_n"].where(S["_n"] > 0, 1), np.nan)
    M["diff_fb"] = np.where(ok, M["fb"] - S["_fb"] / S["_n"].where(S["_n"] > 0, 1), np.nan)
    M["ctrl_fb"] = np.where(ok, S["_fb"] / S["_n"].where(S["_n"] > 0, 1), np.nan)
    m_ok = M[ok]
    pc = m_ok.groupby("client").agg(s_lr=("diff_lr", "sum"), s_fb=("diff_fb", "sum"), n=("diff_lr", "size"))
    est_lr = pc["s_lr"].sum() / pc["n"].sum()
    ci_lr = boot_ci(pc["s_lr"].values, pc["n"].values)
    est_fb = pc["s_fb"].sum() / pc["n"].sum()
    ci_fb = boot_ci(pc["s_fb"].values, pc["n"].values)

    line = {}
    line["pair"] = f"{a} -> {b}"
    line["flagged_in_A_with_matched_controls"] = f"{int(ok.sum())} of {int(fa.sum())} (clients {pc.shape[0]})"
    line["matched_clicks_ratio_flagged_vs_controls"] = f"{np.exp(est_lr):.3f} (95% CI {np.exp(ci_lr[0]):.3f} to {np.exp(ci_lr[1]):.3f})"
    line["flagged_again_in_B_flagged"] = f"{m_ok['fb'].mean():.3f}"
    line["flagged_in_B_matched_controls"] = f"{m_ok['ctrl_fb'].mean():.3f}"
    line["flag_rate_difference_points"] = f"{100*est_fb:.1f} (95% CI {100*ci_fb[0]:.1f} to {100*ci_fb[1]:.1f})"
    cc = M.loc[fa, "client"].value_counts()
    line["flagged_A_clients_and_largest_share"] = f"{len(cc)} clients, largest {cc.iloc[0]/fa.sum():.1%} (largest client's share of all pages: {(M['client']==cc.index[0]).mean():.1%})"

    M["dec"] = pd.qcut(np.log(M["ratio_a"]).rank(method="first"), 10, labels=False)
    line["median_ratio_B_by_decile_of_ratio_A(0=worst)"] = [round(float(x), 3) for x in M.groupby("dec")["ratio_b"].median()]
    M = M.sort_values("missed_a", ascending=False)
    tot = M["missed_b"].sum()
    n = len(M)
    line["share_of_B_missed_clicks_in_top_q_of_A_ranking"] = {f"top {int(q*100)}%": round(float(M["missed_b"].head(int(n*q)).sum() / tot), 3) for q in [0.01, 0.02, 0.05, 0.10]}

    per = M[M["flag_a"] & M["flag_b"]].copy()
    per["band"] = pd.cut(per["pos_b"], [-1, 3.5, 10.5, 1000], labels=["position <=3.5", "position 3.6-10.5", "position >10.5"])
    line["persistent_pages_flagged_both_months"] = f"{len(per)} from {per['client'].nunique()} clients"
    line["persistent_by_position_band(pages, share_of_their_B_missed_clicks)"] = {
        str(k): (int(len(v)), round(float(v['missed_b'].sum() / per['missed_b'].sum()), 3)) for k, v in per.groupby("band", observed=True)}
    per["rk"] = per.groupby("client")["missed_b"].rank(ascending=False, method="first")
    line["persistent_top5_per_client"] = f"{int((per['rk'] <= 5).sum())} pages from {per.loc[per['rk'] <= 5, 'client'].nunique()} clients"
    OUT.append(line)

print("===== COPY EVERYTHING BELOW THIS LINE AND SEND IT TO ME =====")
for line in OUT:
    print()
    for k, v in line.items():
        print(f"{k}: {v}")
#### END PART 2
