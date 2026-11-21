"""Does the leaf-table idea carry over to interventional TreeSHAP (B) and KernelSHAP (C)?
Reference = the explainer's own output on the test case. Corrections for a test inheriting from leader e:
  copy      : phi[e]
  corr_pd   : phi[e] + path-dependent leaf-table difference (free, no extra explainer runs)
  corr_ridge: phi[e] + (z_i - z_e) W, W = ridge fit of the explainer's own output on S training rows
              (z = one-hot of (tree, leaf)); needs S explainer runs per trained model
  tab_ridge : z_i W for every test, no explainer at test time
Final version for the 25-subject run: ridge variants dropped (no gain), additivity check relaxed for interventional
TreeSHAP, KernelSHAP limited to small builds (set KERNEL=0 to skip it).
usage: python3 improve_ik25.py <datasets_dir> <out_dir> <subject...>"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore"); os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, lightgbm as lgb, shap
import run_tcpci as R
from improve_corrected import build_table, corrected

def onehot_dense(leaves):
    n, T = leaves.shape
    Z = np.zeros((n, T * R.LEAVES), dtype=np.float32)
    Z[np.arange(n)[:, None], np.arange(T)[None, :] * R.LEAVES + leaves] = 1
    return Z

def ridge(Z, Phi, lam=1.0):
    K = Z @ Z.T + lam * np.eye(len(Z), dtype=np.float32)
    return Z.T @ np.linalg.solve(K, Phi)

def corr_ridge(phi, Zall, lead, W):
    out = phi[lead].copy()
    idx = np.flatnonzero(lead != np.arange(len(lead)))
    out[idx] += (Zall[idx] - Zall[lead[idx]]) @ W
    return out

def run(ddir, name, out):
    t0 = time.time()
    df = pd.read_csv(os.path.join(ddir, name, "dataset.csv"))
    drop = [c for c in df.columns if c in ("Build", "Test", "Verdict", "Duration") or c.startswith("DET_")]
    feats = [c for c in df.columns if c not in drop]; F = len(feats)
    df["_fail"] = (df["Verdict"] > 0).astype(int)
    builds = np.sort(df["Build"].unique()); nb = len(builds)
    tr_b = builds[: int(nb * R.TRAIN_FRAC)]
    cal_b = builds[int(nb * R.TRAIN_FRAC): int(nb * (R.TRAIN_FRAC + R.CAL_FRAC)) + 1]
    ev_pool = builds[int(nb * (R.TRAIN_FRAC + R.CAL_FRAC)):]
    df = df.sort_values(["Build", "Test"], kind="stable"); grp = df.groupby("Build", sort=True)
    tr = df[df["Build"].isin(tr_b)]
    model = lgb.LGBMRanker(n_estimators=R.TREES, num_leaves=R.LEAVES, learning_rate=0.1, objective="lambdarank",
                           random_state=0, n_jobs=1, verbose=-1)
    model.fit(tr[feats].to_numpy(np.float32), tr["_fail"].to_numpy(), group=tr.groupby("Build", sort=True).size().to_numpy())
    explA = shap.TreeExplainer(model)
    cal = df[df["Build"].isin(cal_b)]; cal = cal[cal.groupby("Build")["Build"].transform("size") >= R.MIN_TESTS]
    if len(cal) > 3000: cal = cal.iloc[:3000]
    if len(cal) < 50: cal = tr.groupby("Build").filter(lambda g: len(g) >= R.MIN_TESTS).tail(1500)
    Xc = cal[feats].to_numpy(np.float32)
    lc, sc, pc = model.predict(Xc, pred_leaf=True), model.predict(Xc), explA.shap_values(Xc)
    tau = 0.95
    for t in np.linspace(0.3, 0.95, 14):
        lo_, _ = R.rolc_assign(lc, sc, t); c = R.cosine_rows(pc, pc[lo_]).mean(); tau = float(t)
        if c >= R.TARGET_COS: break
    rs = np.random.RandomState(0)
    Xtr_all = tr[feats].to_numpy(np.float32)
    bg100 = Xtr_all[rs.choice(len(Xtr_all), min(100, len(Xtr_all)), replace=False)]
    bg50 = bg100[:50]
    explB = shap.TreeExplainer(model, data=bg100, feature_perturbation="interventional")
    explC = shap.KernelExplainer(model.predict, bg50)
    trs = tr.iloc[np.linspace(0, len(tr) - 1, min(1000, len(tr))).astype(int)]
    tab, _ = build_table(model, trs[feats].to_numpy(np.float32), F)
    cand = []
    for b in ev_pool:
        g = grp.get_group(b)
        if R.MIN_TESTS <= len(g) and g["_fail"].sum() >= 1: cand.append((b, len(g)))
    maxb = 3 if name.startswith("JMRI") else 6
    if len(cand) > maxb: cand = [cand[i] for i in np.linspace(0, len(cand) - 1, maxb).astype(int)]
    use_kernel = os.environ.get("KERNEL", "1") != "0"
    rows = []
    for b, n in cand:
        g = grp.get_group(b); X = g[feats].to_numpy(np.float32)
        leaves, scores = model.predict(X, pred_leaf=True), model.predict(X)
        lead, isl = R.rolc_assign(leaves, scores, tau); m = int(isl.sum())
        if m >= n: continue
        r = dict(subject=name, build=int(b), n=n, m=m)
        phiB = explB.shap_values(X, check_additivity=False)
        r["B_copy"] = float(R.cosine_rows(phiB, phiB[lead]).mean())
        r["B_corr"] = float(R.cosine_rows(phiB, corrected(phiB, leaves, lead, tab)).mean())
        r["B_below07_copy"] = float((R.cosine_rows(phiB, phiB[lead]) < .7).mean())
        r["B_below07_corr"] = float((R.cosine_rows(phiB, corrected(phiB, leaves, lead, tab)) < .7).mean())
        if use_kernel and n <= 150:
            phiC = explC.shap_values(X, silent=True)
            r["C_copy"] = float(R.cosine_rows(phiC, phiC[lead]).mean())
            r["C_corr"] = float(R.cosine_rows(phiC, corrected(phiC, leaves, lead, tab)).mean())
        rows.append(r)
        print(f"  {name} b={b} n={n} m={m} interventional: {r['B_copy']:.3f} -> {r['B_corr']:.3f}" + (f" | kernel: {r['C_copy']:.3f} -> {r['C_corr']:.3f}" if "C_copy" in r else ""), flush=True)
    json.dump(dict(subject=name, rows=rows), open(os.path.join(out, name + ".json"), "w"))
    print(f"done {name} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    d, o = sys.argv[1], sys.argv[2]; os.makedirs(o, exist_ok=True)
    for s in sys.argv[3:]:
        try: run(d, s, o)
        except Exception as e: print("FAILED", s, repr(e), flush=True)
