"""
End-to-end (whole-pipeline) runtime and memory benchmark on TCP-CI.

For each evaluation build b of a subject:
  1. retrain LambdaMART on ALL earlier builds (expanding window)      -> t_train
  2. score the build's test cases                                     -> t_score
  3. (reuse pipelines) leaf indices, stored as uint8, chunked         -> t_leaf
  4. (reuse pipelines) choose which test cases get their own explanation -> t_assign
  5. explain: n test cases (individual) or m leaders (reuse)

Three explainer costs, because the saving depends on how expensive one explanation is:
  A  TreeSHAP, path-dependent       (measured directly on the whole set)
  B  TreeSHAP, interventional, 100 background rows  (per-row cost from a sample x count)
  C  KernelSHAP, 50 background rows, default nsamples (per-row cost from 3 rows x count)

Pipelines: Individual, ROLC, CLARA-style K-Medoids, K-Medoids (only if n <= 1500; its
n x n matrix makes larger builds impractical, which is itself the finding).
Memory: tracemalloc peak of steps 3-5 under explainer A on the largest evaluation build.
tracemalloc does not see allocations made inside C extensions, so peak memory of the
TreeSHAP internals is not included; explanation OUTPUT storage is reported analytically too.

usage: python3 e2e.py <datasets_dir> <out_dir> <subject> [subject ...]
"""
import sys, os, time, json, warnings, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb, shap
import run_tcpci as R

NB_EVAL = 5
TREES, LEAVES = R.TREES, R.LEAVES


def leaves_u8(model, X, blk=512):
    n = len(X); T = model.n_estimators
    out = np.empty((n, T), np.uint8)
    for s in range(0, n, blk):
        out[s:s + blk] = model.predict(X[s:s + blk], pred_leaf=True)
    return out


def peak_mb(f):
    tracemalloc.start(); f(); _, p = tracemalloc.get_traced_memory(); tracemalloc.stop(); return p / 1e6


def run_subject(ddir, name, out):
    t0 = time.time()
    df = pd.read_csv(os.path.join(ddir, name, "dataset.csv"))
    drop = [c for c in df.columns if c in ("Build", "Test", "Verdict", "Duration") or c.startswith("DET_")]
    feats = [c for c in df.columns if c not in drop]
    df["_fail"] = (df["Verdict"] > 0).astype(int)
    df = df.sort_values(["Build", "Test"], kind="stable")
    builds = np.sort(df["Build"].unique()); nb = len(builds)
    pool = builds[int(nb * 0.60):]
    grp = df.groupby("Build", sort=True)
    cand = [(b, len(grp.get_group(b))) for b in pool
            if len(grp.get_group(b)) >= R.MIN_TESTS and grp.get_group(b)["_fail"].sum() >= 1]
    nev = 2 if name.startswith("JMRI") else NB_EVAL
    if len(cand) > nev:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, nev).astype(int)]
    big = max(cand, key=lambda x: x[1])[0] if cand else None

    rows = []
    for b, n in cand:
        hist = df[df["Build"] < b]
        Xh = hist[feats].to_numpy(np.float32); yh = hist["_fail"].to_numpy()
        gs = hist.groupby("Build", sort=True).size().to_numpy()
        model = lgb.LGBMRanker(n_estimators=TREES, num_leaves=LEAVES, learning_rate=0.1,
                               objective="lambdarank", random_state=0, n_jobs=1, verbose=-1)
        _, t_train = R.timed(lambda: model.fit(Xh, yh, group=gs))
        g = grp.get_group(b)
        X = g[feats].to_numpy(np.float32)
        scores, t_score = R.timed(lambda: model.predict(X))
        leaves, t_leaf = R.timed(lambda: leaves_u8(model, X))

        # tau calibrated on the last 3 training builds (held-out from nothing: cheap proxy)
        cal_b = np.sort(hist["Build"].unique())[-3:]
        cal = hist[hist["Build"].isin(cal_b)].iloc[:2500]
        Xc = cal[feats].to_numpy(np.float32)
        explA = shap.TreeExplainer(model)
        lc, sc, pc = leaves_u8(model, Xc), model.predict(Xc), explA.shap_values(Xc)
        tau = 0.95
        for t in np.linspace(0.3, 0.95, 14):
            lo, _ = R.rolc_assign(lc, sc, t)
            if R.cosine_rows(pc, pc[lo]).mean() >= R.TARGET_COS:
                tau = float(t); break

        # A: TreeSHAP path-dependent, measured directly
        phi, tA_ind = R.timed(lambda: explA.shap_values(X))
        (lo, isl), t_rolc = R.timed(lambda: R.rolc_assign(leaves, scores, tau))
        m = int(isl.sum())
        if m >= n: continue
        _, tA_m = R.timed(lambda: explA.shap_values(X[isl]))
        t_clara_lead, t_clara = R.timed(lambda: R.clara_assign(leaves, m))
        t_kmed = np.nan
        if n <= 1500:
            _, t_kmed = R.timed(lambda: R.kmedoids_assign(leaves, m))

        # B: interventional TreeSHAP, per-row cost
        rng = np.random.default_rng(0)
        bg = Xh[rng.choice(len(Xh), min(100, len(Xh)), replace=False)]
        explB = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        s = min(60, n)
        _, tB = R.timed(lambda: explB.shap_values(X[:s], check_additivity=False))
        cB = tB / s
        # C: KernelSHAP per-row cost
        bg50 = shap.kmeans(bg, 10) if len(bg) >= 10 else bg
        explC = shap.KernelExplainer(model.predict, bg50)
        s3 = min(3, n)
        _, tC = R.timed(lambda: explC.shap_values(X[:s3], silent=True))
        cC = tC / s3

        r = dict(subject=name, build=int(b), n=n, m=m, tau=tau, n_hist_rows=int(len(Xh)),
                 t_train=t_train, t_score=t_score, t_leaf=t_leaf,
                 t_assign_ROLC=t_rolc, t_assign_CLARA=t_clara, t_assign_KMedoids=t_kmed,
                 expA_ind=tA_ind, expA_m=tA_m, cB=cB, cC=cC)
        common = t_train + t_score
        for ex, ind_cost, cost_m in (("A", tA_ind, tA_m), ("B", n * cB, m * cB), ("C", n * cC, m * cC)):
            r[f"{ex}_total_Individual"] = common + ind_cost
            for meth, ta in (("ROLC", t_rolc), ("CLARA", t_clara), ("KMedoids", t_kmed)):
                r[f"{ex}_total_{meth}"] = common + t_leaf + ta + cost_m
            r[f"{ex}_expl_share_Individual"] = ind_cost / (common + ind_cost)

        if b == big:
            r["mem_Individual_mb"] = peak_mb(lambda: explA.shap_values(X))
            def reuse(method):
                lv = leaves_u8(model, X)
                if method == "ROLC":
                    _, il = R.rolc_assign(lv, scores, tau)
                else:
                    ld = R.clara_assign(lv, m) if method == "CLARA" else R.kmedoids_assign(lv, m)
                    il = np.zeros(n, bool); il[np.unique(ld)] = True
                explA.shap_values(X[il])
            r["mem_ROLC_mb"] = peak_mb(lambda: reuse("ROLC"))
            r["mem_CLARA_mb"] = peak_mb(lambda: reuse("CLARA"))
            if n <= 1500:
                r["mem_KMedoids_mb"] = peak_mb(lambda: reuse("KMedoids"))
            p = X.shape[1]
            r["out_bytes_Individual"] = n * p * 8
            r["out_bytes_reuse"] = m * p * 8 + n * (4) + n * TREES  # SHAP rows + leader index + uint8 leaves
        rows.append(r)
        print(f"  {name} b={b} n={n} m={m} train={t_train:.2f}s", flush=True)
    json.dump(dict(subject=name, rows=rows, secs=time.time() - t0),
              open(os.path.join(out, name + ".json"), "w"), indent=1)
    print(f"done {name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ddir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for s in sys.argv[3:]:
        try:
            run_subject(ddir, s, out)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", s, repr(e), flush=True)
