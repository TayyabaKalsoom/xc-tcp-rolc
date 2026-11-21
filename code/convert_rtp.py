"""Convert RTPTorrent project CSVs into TCP-CI-style dataset.csv (Build, Test, Verdict, Duration + history features).
Features use ONLY earlier jobs of the same test class (shifted), so there is no look-ahead.
Jobs without a failing test carry no ranking signal for LambdaMART, so only jobs with >=MIN_TESTS tests and
>=1 failing test are kept (capped at MAXJ evenly spaced) AFTER features are computed on the full history."""
import sys, os, numpy as np, pandas as pd
MIN_TESTS, MAXJ = 30, 1500
src, dst = sys.argv[1], sys.argv[2]
def convert(proj):
    d = pd.read_csv(f"{src}/{proj}/{proj}.csv")
    d["fail"] = ((d.failures > 0) | (d.errors > 0)).astype(int)
    d = (d.groupby(["travisJobId", "testName"], sort=False)
           .agg(fail=("fail", "max"), duration=("duration", "sum"), count=("count", "sum"), index=("index", "min")).reset_index())
    d = d.sort_values(["testName", "travisJobId"], kind="stable").reset_index(drop=True)
    d["Test"] = pd.factorize(d.testName)[0]
    g = d.groupby("Test", sort=False)
    v = d["fail"].astype(float); pos = g.cumcount().astype(float)
    prev_v = g["fail"].shift(1)
    tr = (v != prev_v).astype(float).where(prev_v.notna())
    f = pd.DataFrame({"Test": d["Test"]})
    f["REC_Age"] = pos
    f["REC_LastVerdict"] = prev_v.fillna(-1)
    lf = pos.where(v == 1).groupby(d["Test"]).ffill().groupby(d["Test"]).shift(1)
    f["REC_LastFailureAge"] = (pos - lf).fillna(-1)
    lt = pos.where(tr == 1).groupby(d["Test"]).ffill().groupby(d["Test"]).shift(1)
    f["REC_LastTransitionAge"] = (pos - lt).fillna(-1)
    sh = lambda s, fn: s.groupby(d["Test"]).transform(lambda x: fn(x.shift(1)))
    f["REC_RecentFailRate"] = sh(v, lambda x: x.rolling(5, min_periods=1).mean()).fillna(-1)
    f["REC_TotalFailRate"] = ((v.groupby(d["Test"]).cumsum() - v) / pos.where(pos > 0)).fillna(-1)
    trf = tr.fillna(0)
    f["REC_RecentTransitionRate"] = sh(trf, lambda x: x.rolling(5, min_periods=1).mean()).fillna(-1)
    f["REC_TotalTransitionRate"] = ((trf.groupby(d["Test"]).cumsum()) / pos.where(pos > 0)).fillna(-1)
    du = d["duration"].astype(float)
    f["REC_LastExeTime"] = g["duration"].shift(1).fillna(-1)
    f["REC_RecentAvgExeTime"] = sh(du, lambda x: x.rolling(5, min_periods=1).mean()).fillna(-1)
    f["REC_RecentMaxExeTime"] = sh(du, lambda x: x.rolling(5, min_periods=1).max()).fillna(-1)
    f["REC_TotalAvgExeTime"] = ((du.groupby(d["Test"]).cumsum() - du) / pos.where(pos > 0)).fillna(-1)
    f["REC_TotalMaxExeTime"] = sh(du, lambda x: x.expanding().max()).fillna(-1)
    f["REC_LastCount"] = g["count"].shift(1).fillna(-1)
    f["REC_LastIndex"] = g["index"].shift(1).fillna(-1)
    out = pd.concat([d[["travisJobId", "Test"]].rename(columns={"travisJobId": "Build"}),
                     d["fail"].rename("Verdict"), d["duration"].rename("Duration"), f.drop(columns="Test")], axis=1)
    js = out.groupby("Build").agg(n=("Test", "size"), nf=("Verdict", "sum"))
    keep = js[(js.n >= MIN_TESTS) & (js.nf >= 1)].index.to_numpy()
    if len(keep) > MAXJ:
        keep = keep[np.linspace(0, len(keep) - 1, MAXJ).astype(int)]
    out = out[out["Build"].isin(keep)].sort_values(["Build", "Test"], kind="stable")
    os.makedirs(f"{dst}/{proj}", exist_ok=True)
    out.to_csv(f"{dst}/{proj}/dataset.csv", index=False)
    return proj, len(keep), len(out), out.Build.nunique()
if __name__ == "__main__":
    skip = {"adamfisk@LittleProxy", "deeplearning4j@deeplearning4j"}  # too few failing jobs with >=30 tests
    for p in sorted(os.listdir(src)):
        if not os.path.isdir(f"{src}/{p}") or p in skip: continue
        print(convert(p), flush=True)
