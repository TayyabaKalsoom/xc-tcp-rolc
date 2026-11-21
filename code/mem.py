"""Memory-lean ROLC and a fair memory comparison (tracemalloc peak, explanation phase only)."""
import sys, os, json, time, warnings, tracemalloc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, shap
import run_tcpci as R
from e2e import leaves_u8

def rolc_lean(leaves, scores, tau, refine=True):
    """Same decisions as R.rolc_assign, minus the float temporaries and the full-size leader copy."""
    n, T = leaves.shape
    order = np.argsort(-scores, kind="stable")
    leader_of = np.full(n, -1, np.int32)
    ids = np.empty(64, np.int32); L = np.empty((64, T), leaves.dtype); m = 0
    for i in order:
        row = leaves[i]
        if m:
            cnt = np.count_nonzero(L[:m] == row, axis=1)
            j = int(cnt.argmax())
            if cnt[j] >= tau * T - 1e-9:
                leader_of[i] = ids[j]; continue
        if m == len(ids):
            ids = np.concatenate([ids, np.empty(m, np.int32)]); L = np.concatenate([L, np.empty_like(L)])
        L[m] = row; ids[m] = i; leader_of[i] = i; m += 1
    if refine and m > 1:
        Lm = L[:m]; isl = np.zeros(n, bool); isl[ids[:m]] = True
        for i in np.flatnonzero(~isl):
            leader_of[i] = ids[int(np.count_nonzero(Lm == leaves[i], axis=1).argmax())]
    return leader_of, ids[:m].copy()

def explain_ids(ex, X, ids, blk=256):
    out = np.empty((len(ids), X.shape[1]))
    for s in range(0, len(ids), blk):
        out[s:s + blk] = ex.shap_values(X[ids[s:s + blk]])
    return out

def peak(f):
    tracemalloc.start(); f(); _, p = tracemalloc.get_traced_memory(); tracemalloc.stop(); return p / 1e6

def main(ddir, out):
    rows = []
    for name in sorted(os.listdir(ddir)):
        if not os.path.exists(os.path.join(ddir, name, "dataset.csv")): continue
        df = pd.read_csv(os.path.join(ddir, name, "dataset.csv"))
        drop = [c for c in df.columns if c in ("Build", "Test", "Verdict", "Duration") or c.startswith("DET_")]
        feats = [c for c in df.columns if c not in drop]
        df["_fail"] = (df["Verdict"] > 0).astype(int); df = df.sort_values(["Build", "Test"], kind="stable")
        builds = np.sort(df["Build"].unique()); nb = len(builds)
        tr = df[df["Build"].isin(builds[: int(nb * 0.55)])]
        model = lgb.LGBMRanker(n_estimators=R.TREES, num_leaves=R.LEAVES, learning_rate=0.1, objective="lambdarank", random_state=0, n_jobs=1, verbose=-1)
        model.fit(tr[feats].to_numpy(np.float32), tr["_fail"].to_numpy(), group=tr.groupby("Build", sort=True).size().to_numpy())
        ex = shap.TreeExplainer(model)
        grp = df.groupby("Build", sort=True)
        pool = [b for b in builds[int(nb * 0.6):] if len(grp.get_group(b)) >= R.MIN_TESTS and grp.get_group(b)["_fail"].sum() >= 1]
        b = max(pool, key=lambda x: len(grp.get_group(x)))
        X = grp.get_group(b)[feats].to_numpy(np.float32); n = len(X)
        lv0 = leaves_u8(model, X); sc = model.predict(X)
        # tau as in the main runs (calibrate quickly on the 3 builds before the evaluation pool)
        cal = df[df["Build"].isin(builds[int(nb * .55): int(nb * .6) + 1])].iloc[:2500]
        if len(cal) < 50: cal = tr.tail(1500)
        Xc = cal[feats].to_numpy(np.float32); lc, sc_c, pc = leaves_u8(model, Xc), model.predict(Xc), ex.shap_values(Xc)
        tau = 0.95
        for t in np.linspace(0.3, 0.95, 14):
            lo, _ = R.rolc_assign(lc, sc_c, t)
            if R.cosine_rows(pc, pc[lo]).mean() >= R.TARGET_COS: tau = float(t); break
        old_lead, old_isl = R.rolc_assign(lv0, sc, tau)
        new_lead, ids = rolc_lean(lv0, sc, tau)
        same = bool(np.array_equal(old_lead, new_lead)); m = len(ids)
        t0 = time.perf_counter(); rolc_lean(lv0, sc, tau); t_lean = time.perf_counter() - t0
        t0 = time.perf_counter(); R.rolc_assign(lv0, sc, tau); t_old = time.perf_counter() - t0
        r = dict(subject=name, n=n, m=m, same_decisions=same, t_assign_old=t_old, t_assign_lean=t_lean,
                 mem_individual=peak(lambda: ex.shap_values(X)),
                 mem_individual_chunked=peak(lambda: explain_ids(ex, X, np.arange(n))),
                 mem_rolc_old=peak(lambda: (lambda lv: (R.rolc_assign(lv, sc, tau), ex.shap_values(X[np.flatnonzero(R.rolc_assign(lv, sc, tau)[1])])))(leaves_u8(model, X))),
                 mem_rolc_lean=peak(lambda: (lambda lv: (lambda r: explain_ids(ex, X, r[1]))(rolc_lean(lv, sc, tau)))(leaves_u8(model, X))))
        rows.append(r); print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "subject"}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
