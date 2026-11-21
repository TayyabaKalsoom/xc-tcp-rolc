"""Test the leaf-table correction for better reused explanations (path-dependent TreeSHAP as reference):
 (1) leaf-table correction: leader's SHAP + sum over trees where member's leaf differs of
     (table[t,leaf_member] - table[t,leaf_leader]); table = mean per-tree SHAP contribution per leaf, learned on
     TRAINING rows only.   Also 'table_only': every test explained from the table, no SHAP at all.
 (2) tree-weighted agreement for ROLC (weights = typical per-tree contribution magnitude), compared at equal m.
 Also reports top-3 feature overlap as a reader-oriented fidelity measure.
usage: python3 improve.py <datasets_dir> <out_dir> <max_builds> <subject...>"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore"); os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, lightgbm as lgb, shap
import run_tcpci as R

def top3(phi): return np.argsort(-np.abs(phi), axis=1)[:, :3]
def overlap(A, B):
    a, b = top3(A), top3(B)
    return np.array([len(set(x) & set(y)) / 3 for x, y in zip(a, b)])

def build_table(model, Xs, F):
    bst = model.booster_; T = model.n_estimators
    lv = model.predict(Xs, pred_leaf=True)
    tab = np.zeros((T, R.LEAVES, F), dtype=np.float32); cnt = np.zeros((T, R.LEAVES))
    for t in range(T):
        c = bst.predict(Xs, start_iteration=t, num_iteration=1, pred_contrib=True)[:, :F]
        for l in np.unique(lv[:, t]):
            msk = lv[:, t] == l
            tab[t, l] = c[msk].mean(0); cnt[t, l] = msk.sum()
    return tab, cnt

def table_phi(tab, leaves):
    T = leaves.shape[1]; ar = np.arange(T)
    return np.stack([tab[ar, leaves[i]].sum(0) for i in range(len(leaves))])

def rolc_w(leaves, scores, tau, w):
    n, T = leaves.shape
    order = np.argsort(-scores, kind="stable")
    leader_of = np.full(n, -1); is_leader = np.zeros(n, bool)
    L = np.empty((n, T), dtype=leaves.dtype); ids = np.empty(n, dtype=np.int64); m = 0
    for i in order:
        if m:
            agree = ((L[:m] == leaves[i]) * w).sum(1)
            j = int(agree.argmax())
            if agree[j] >= tau: leader_of[i] = ids[j]; continue
        L[m] = leaves[i]; ids[m] = i; leader_of[i] = i; is_leader[i] = True; m += 1
    for i in np.flatnonzero(~is_leader):
        leader_of[i] = ids[int((((L[:m] == leaves[i]) * w).sum(1)).argmax())]
    return leader_of, is_leader

def match_m(leaves, scores, w, m_target):
    lo, hi = 0.05, 0.999; best = None
    for _ in range(9):
        mid = (lo + hi) / 2
        lead, isl = rolc_w(leaves, scores, mid, w); m = int(isl.sum())
        if best is None or abs(m - m_target) < abs(best[1] - m_target): best = (lead, m)
        if m > m_target: hi = mid
        else: lo = mid
    return best

def corrected(phi, leaves, lead, tab):
    out = phi[lead].copy(); T = leaves.shape[1]; ar = np.arange(T)
    for i in np.flatnonzero(lead != np.arange(len(lead))):
        e = lead[i]; d = leaves[i] != leaves[e]
        if d.any(): out[i] += tab[ar[d], leaves[i][d]].sum(0) - tab[ar[d], leaves[e][d]].sum(0)
    return out

def run(ddir, name, out, maxb):
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
    expl = shap.TreeExplainer(model)
    cal = df[df["Build"].isin(cal_b)]; cal = cal[cal.groupby("Build")["Build"].transform("size") >= R.MIN_TESTS]
    if len(cal) > 3000: cal = cal.iloc[:3000]
    if len(cal) < 50: cal = tr.groupby("Build").filter(lambda g: len(g) >= R.MIN_TESTS).tail(1500)
    Xc = cal[feats].to_numpy(np.float32)
    lc, sc, pc = model.predict(Xc, pred_leaf=True), model.predict(Xc), expl.shap_values(Xc)
    tau = 0.95
    for t in np.linspace(0.3, 0.95, 14):
        lo_, _ = R.rolc_assign(lc, sc, t); c = R.cosine_rows(pc, pc[lo_]).mean(); tau = float(t)
        if c >= R.TARGET_COS: break
    # leaf table from TRAINING rows only (sample up to 4000, spread over builds)
    trs = tr.iloc[np.linspace(0, len(tr) - 1, min(int(os.environ.get("TABLE_ROWS", 1000)), len(tr))).astype(int)]
    t_ = time.perf_counter(); tab, cnt = build_table(model, trs[feats].to_numpy(np.float32), F); ttab = time.perf_counter() - t_
    cand = []
    for b in ev_pool:
        g = grp.get_group(b)
        if R.MIN_TESTS <= len(g) and g["_fail"].sum() >= 1: cand.append((b, len(g)))
    if name.startswith("JMRI"): maxb = min(maxb, 3)
    if len(cand) > maxb: cand = [cand[i] for i in np.linspace(0, len(cand) - 1, maxb).astype(int)]
    rows = []
    for b, n in cand:
        g = grp.get_group(b); X = g[feats].to_numpy(np.float32)
        leaves, scores = model.predict(X, pred_leaf=True), model.predict(X)
        t_ = time.perf_counter(); phi = expl.shap_values(X); tphi = time.perf_counter() - t_
        lead, isl = R.rolc_assign(leaves, scores, tau); m = int(isl.sum())
        if m >= n: continue
        c0 = R.cosine_rows(phi, phi[lead])
        t = time.perf_counter(); ph1 = corrected(phi, leaves, lead, tab); tcorr = time.perf_counter() - t
        c1 = R.cosine_rows(phi, ph1)
        tp = table_phi(tab, leaves); ct = R.cosine_rows(phi, tp)
        if n <= 1500:
            km = R.kmedoids_assign(leaves, m); ck = R.cosine_rows(phi, phi[km])
        else:
            km = lead; ck = np.full(n, np.nan)
        rows.append(dict(subject=name, build=int(b), n=n, m=m, t_table_build=ttab,
            cos_rolc=float(c0.mean()), cos_rolc_corr=float(c1.mean()), cos_table_only=float(ct.mean()), t_phi_individual=tphi,
            cos_kmed=float(np.nanmean(ck)), cos_kmed_corr=(float(R.cosine_rows(phi, corrected(phi, leaves, km, tab)).mean()) if n <= 1500 else float('nan')),
            b07_rolc=float((c0 < .7).mean()), b07_rolc_corr=float((c1 < .7).mean()),
            top3_rolc=float(overlap(phi, phi[lead]).mean()), top3_rolc_corr=float(overlap(phi, ph1).mean()),
            top3_table=float(overlap(phi, tp).mean()), top3_kmed=float(overlap(phi, phi[km]).mean()) if n <= 1500 else float("nan"),
            t_corr=tcorr, t_ind=0.0))
        print(f"  {name} b={b} n={n} m={m} cos {c0.mean():.3f} -> corr {c1.mean():.3f} | table {ct.mean():.3f} | kmed {ck.mean():.3f}", flush=True)
    json.dump(dict(subject=name, rows=rows), open(os.path.join(out, name + ".json"), "w"))
    print(f"done {name} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    d, o, mb = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(o, exist_ok=True)
    for s in sys.argv[4:]:
        try: run(d, s, o, mb)
        except Exception as e: print("FAILED", s, repr(e), flush=True)
