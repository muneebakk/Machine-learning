# CTR scoring notebook 3b, ALL IN ONE CELL.
# Paste this whole file into ONE Colab cell and press Run. Needs the Colab secret HF_TOKEN (Notebook access ON).

# 1. Setup, load, same clean analysis set as notebook 3. Needs Colab secret HF_TOKEN.
!pip install -q datasets huggingface_hub
from google.colab import userdata
from huggingface_hub import login
login(userdata.get("HF_TOKEN"))

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import poisson, norm

ds = load_dataset("FlyRank/internship-lanes", "engagement_fix")
df = ds["train"].to_pandas()

base = df[(df["impressions_30d"] > 0) & df["avg_position_30d"].notna()]
d = base[base["impressions_30d"] >= 100].copy()
d["ctr"] = d["clicks_30d"] / d["impressions_30d"]
bins = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 10.5, 15.5, 20.5, 1000]
d["bucket"] = pd.cut(d["avg_position_30d"], bins=bins, labels=False)
glob = d.groupby("bucket")[["clicks_30d", "impressions_30d"]].sum()
d["global_ctr"] = d["bucket"].map(glob["clicks_30d"] / glob["impressions_30d"])
cb = d.groupby(["client_hash_id", "bucket"])[["clicks_30d", "impressions_30d"]].transform("sum")
d["loo_imp"] = cb["impressions_30d"] - d["impressions_30d"]
d["loo_clk"] = cb["clicks_30d"] - d["clicks_30d"]
d["loo_ctr"] = np.where(d["loo_imp"] > 0, d["loo_clk"] / d["loo_imp"].where(d["loo_imp"] > 0, 1), 0.0)
print(f"{len(d):,} pages, {d['client_hash_id'].nunique()} clients")

def expectation(K):
    wgt = d["loo_imp"] / (d["loo_imp"] + K)
    exp_ctr = np.clip(wgt * d["loo_ctr"] + (1 - wgt) * d["global_ctr"], 1e-6, 1)
    return exp_ctr, wgt

def flag_poisson(exp_ctr):
    exp_clicks = exp_ctr * d["impressions_30d"]
    p = poisson.cdf(d["clicks_30d"], exp_clicks)
    missed = (exp_clicks - d["clicks_30d"]).clip(lower=0)
    return (p < 0.01) & (missed >= 5), missed

# 2. Try small, medium and large K
rows = []
ref_top = None
for K in [5_000, 50_000, 500_000, 5_000_000]:
    e, wgt = expectation(K)
    f, missed = flag_poisson(e)
    top = set(missed.nlargest(100).index)
    if ref_top is None: ref_top = top
    rows.append(dict(K=K, median_client_weight=round(float(wgt.median()), 3), flagged=int(f.sum()),
                     flagged_pct=round(100 * f.mean(), 1), top100_overlap_with_K5000=round(100 * len(top & ref_top) / 100)))
print(pd.DataFrame(rows).to_string(index=False))

# 3. Dispersion and quasi-Poisson flag, using K = 50,000
K = 50_000
d["expected_ctr"], wgt = expectation(K)
d["exp_clicks"] = d["expected_ctr"] * d["impressions_30d"]
d["pearson"] = (d["clicks_30d"] - d["exp_clicks"]) / np.sqrt(d["exp_clicks"])
phi = float(np.median(d["pearson"] ** 2) / 0.4549)   # 0.4549 = median of chi-square(1)
d["p_low_pois"] = poisson.cdf(d["clicks_30d"], d["exp_clicks"])
d["p_low_qp"] = norm.cdf(d["pearson"] / np.sqrt(max(phi, 1.0)))
d["missed_clicks"] = (d["exp_clicks"] - d["clicks_30d"]).clip(lower=0)

f_pois = (d["p_low_pois"] < 0.01) & (d["missed_clicks"] >= 5)
f_qp = (d["p_low_qp"] < 0.01) & (d["missed_clicks"] >= 5)
print(f"Dispersion phi = {phi:.1f} (1.0 would mean pure Poisson noise)")
print(f"Flagged, plain Poisson   : {f_pois.sum():,} ({f_pois.mean():.1%})")
print(f"Flagged, overdispersion-adjusted: {f_qp.sum():,} ({f_qp.mean():.1%})")
print(f"Pages flagged by both: {(f_pois & f_qp).sum():,}")
print(f"Adjusted flags hold {d.loc[f_qp, 'missed_clicks'].sum() / d['missed_clicks'].sum():.1%} of all estimated missed clicks")

# 4. Outlier dependence
flag = d[f_qp].copy().sort_values("missed_clicks", ascending=False)
tot = flag["missed_clicks"].sum()
for n in [1, 10, 100]:
    print(f"Top {n:>3} pages = {flag['missed_clicks'].head(n).sum() / tot:.1%} of flagged missed clicks")

extreme = flag[(flag["clicks_30d"] < 0.05 * flag["exp_clicks"]) & (flag["exp_clicks"] >= 1000)]
print(f"\nExtreme cases (under 5% of expected clicks, expected >= 1000): {len(extreme)} pages, "
      f"{extreme['missed_clicks'].sum() / tot:.1%} of flagged missed clicks")
print("These are better treated as 'verify the data or query mix first' than 'rewrite the title'.")

cc = flag["client_hash_id"].value_counts()
print(f"\nFlagged pages come from {flag['client_hash_id'].nunique()} clients; largest = {cc.iloc[0] / len(flag):.1%} "
      f"(its share of all kept pages: {(d['client_hash_id'] == cc.index[0]).mean():.1%})")

# 5. Robust playbook with an explicit "verify first" category
def action(r):
    if r["clicks_30d"] < 0.05 * r["exp_clicks"] and r["exp_clicks"] >= 1000:
        return "Verify data and query mix first (clicks under 5% of expected)"
    p = r["avg_position_30d"]
    if p <= 3.5:
        return "Check SERP layout and query mix, then review title/meta"
    if p <= 10.5:
        return "Review title and meta description (striking distance)"
    return "Monitor (low position)"

flag["suggested_review"] = flag.apply(action, axis=1)
flag["rank_in_client"] = flag.groupby("client_hash_id")["missed_clicks"].rank(ascending=False, method="first")
playbook = flag[flag["rank_in_client"] <= 5]
print(f"Playbook: {len(playbook)} pages from {playbook['client_hash_id'].nunique()} clients")
print(playbook["suggested_review"].value_counts().to_string())
print()
show = ["content_hash_id", "avg_position_30d", "impressions_30d", "ctr", "expected_ctr", "missed_clicks", "p_low_qp", "suggested_review"]
print("Top 10 (local use only, do NOT publish IDs):")
print(playbook[show].head(10).round(4).to_string(index=False))
print(f"\nneeds_ctr_fix among adjusted-flagged: {flag['needs_ctr_fix'].mean():.3f} vs {d['needs_ctr_fix'].mean():.3f} overall (sanity check only)")

# 6. Numbers to copy back to me (no IDs, no client info)
print("===== COPY EVERYTHING BELOW THIS LINE AND SEND IT TO ME =====")
print(pd.DataFrame(rows).to_string(index=False))
print(f"phi {phi:.1f} | flagged Poisson {f_pois.sum()} ({f_pois.mean():.1%}) | flagged adjusted {f_qp.sum()} ({f_qp.mean():.1%}) | both {(f_pois & f_qp).sum()}")
print(f"Adjusted flags hold {d.loc[f_qp, 'missed_clicks'].sum() / d['missed_clicks'].sum():.1%} of all estimated missed clicks")
for n in [1, 10, 100]:
    print(f"Top {n} = {flag['missed_clicks'].head(n).sum() / tot:.1%} of flagged missed clicks")
print(f"Extreme cases: {len(extreme)} pages, {extreme['missed_clicks'].sum() / tot:.1%} of flagged missed clicks")
print(f"Flagged from {flag['client_hash_id'].nunique()} clients; largest {cc.iloc[0] / len(flag):.1%} (its share of kept pages {(d['client_hash_id'] == cc.index[0]).mean():.1%})")
print(f"Playbook {len(playbook)} pages / {playbook['client_hash_id'].nunique()} clients")
print(playbook["suggested_review"].value_counts().to_string())
print(f"needs_ctr_fix among flagged {flag['needs_ctr_fix'].mean():.3f} vs {d['needs_ctr_fix'].mean():.3f}")
