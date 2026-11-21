"""
Ablation that isolates the representativeness check (ranking held fixed, equal explanation budget).

Framework being dissected (XC-TCP style):
  (1) cluster the build's test cases with K-Medoids on tree-path distance, k0 = round(n * frac)
  (2) every member inherits its medoid's explanation
  (3) CHECK: members whose tree-path agreement with their medoid is < tau are flagged and
      explained individually.
Every variant ranks each test case by its own LambdaMART score, so APFD is identical for all of
them; variants differ ONLY in which test cases get their own explanation and who borrows from whom.
Fidelity = cosine(own TreeSHAP vector, vector actually shown for that test case).

Variants (B = k0 + number flagged by the check = total explanations of "check"):
  individual      every test explained                                  (reference, cos = 1)
  cluster_only    k0 medoids, no check                                  (budget k0 < B)
  check           k0 medoids + members with agreement < tau             (budget B)  <- full method
  random_flag     k0 medoids + the SAME NUMBER of random members        (budget B)  control 1
  oracle_flag     k0 medoids + the same number of members whose borrowed explanation is
                  truly worst (uses the true SHAP vectors; not deployable) (budget B)  upper bound
  more_medoids    K-Medoids with k = B, no check                        (budget B)  control 2
  rolc            rank-ordered leader coverage with tau chosen so that m ~ B (budget ~B)
The check is useful only if it beats random_flag and more_medoids at the same budget.

usage: python3 ablation.py <datasets_dir> <out_dir> <frac> <subject> [subject ...]
"""
import sys, os, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb, shap, kmedoids
import run_tcpci as R
from improve_corrected import build_table, corrected

MAX_N = 1500


def stats(phi, lead):
    c = R.cosine_rows(phi, phi[lead])
    return dict(expl=int(len(np.unique(lead))), cos=float(c.mean()), b07=float((c < 0.7).mean()),
                p10=float(np.percentile(c, 10)))


def rolc_for_budget(leaves, scores, B, lo=0.05, hi=0.999):
    best = None
    for _ in range(9):
        mid = (lo + hi) / 2
        lead, isl = R.rolc_assign(leaves, scores, mid)
        m = int(isl.sum())
        if best is None or abs(m - B) < abs(best[1] - B):
            best = (lead, m)
        if m > B: hi = mid
        else: lo = mid
    return best


def run_subject(ddir, name, out, frac):
    t0 = time.time()
    df = pd.read_csv(os.path.join(ddir, name, "dataset.csv"))
    drop = [c for c in df.columns if c in ("Build", "Test", "Verdict", "Duration") or c.startswith("DET_")]
    feats = [c for c in df.columns if c not in drop]
    df["_fail"] = (df["Verdict"] > 0).astype(int)
    df = df.sort_values(["Build", "Test"], kind="stable")
    builds = np.sort(df["Build"].unique()); nb = len(builds)
    tr_b = builds[: int(nb * R.TRAIN_FRAC)]
    cal_b = builds[int(nb * R.TRAIN_FRAC): int(nb * (R.TRAIN_FRAC + R.CAL_FRAC)) + 1]
    pool = builds[int(nb * (R.TRAIN_FRAC + R.CAL_FRAC)):]
    grp = df.groupby("Build", sort=True)
    tr = df[df["Build"].isin(tr_b)]
    model = lgb.LGBMRanker(n_estimators=R.TREES, num_leaves=R.LEAVES, learning_rate=0.1, objective="lambdarank",
                           random_state=0, n_jobs=1, verbose=-1)
    model.fit(tr[feats].to_numpy(np.float32), tr["_fail"].to_numpy(), group=tr.groupby("Build", sort=True).size().to_numpy())
    expl = shap.TreeExplainer(model)
    trs = tr.iloc[np.linspace(0, len(tr) - 1, min(1000, len(tr))).astype(int)]
    tab, _ = build_table(model, trs[feats].to_numpy(np.float32), len(feats))

    cal = df[df["Build"].isin(cal_b)]
    cal = cal[cal.groupby("Build")["Build"].transform("size") >= R.MIN_TESTS].iloc[:3000]
    if len(cal) < 50:
        cal = tr.groupby("Build").filter(lambda g: len(g) >= R.MIN_TESTS).tail(1500)
    Xc = cal[feats].to_numpy(np.float32)
    lc, sc, pc = model.predict(Xc, pred_leaf=True), model.predict(Xc), expl.shap_values(Xc)
    tau = 0.95
    for t in np.linspace(0.3, 0.95, 14):
        lo, _ = R.rolc_assign(lc, sc, t)
        if R.cosine_rows(pc, pc[lo]).mean() >= R.TARGET_COS:
            tau = float(t); break

    cand = [b for b in pool if R.MIN_TESTS <= len(grp.get_group(b)) <= MAX_N and grp.get_group(b)["_fail"].sum() >= 1]
    if len(cand) > 8:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, 8).astype(int)]
    rng = np.random.default_rng(0)
    rows = []
    for b in cand:
        g = grp.get_group(b)
        X = g[feats].to_numpy(np.float32)
        n = len(X)
        leaves, scores, phi = model.predict(X, pred_leaf=True), model.predict(X), expl.shap_values(X)
        k0 = int(max(2, round(n * frac)))
        D = R.leaf_dist(leaves)
        res = kmedoids.fastpam1(D, k0, max_iter=100, init="build", random_state=0)
        med = np.asarray(res.medoids).astype(np.int64); lab = np.asarray(res.labels).astype(np.int64)
        base = med[lab]
        idx = np.arange(n)
        is_med = np.zeros(n, bool); is_med[med] = True
        agree = 1.0 - D[idx, base]
        flagged = (agree < tau) & ~is_med
        F = int(flagged.sum()); B = k0 + F
        if B >= n or F == 0:  # degenerate: nothing to compare
            continue
        members = np.flatnonzero(~is_med)
        v = {}
        v["individual"] = dict(expl=n, cos=1.0, b07=0.0, p10=1.0)
        rf = np.zeros(n, bool); rf[rng.choice(members, F, replace=False)] = True
        true_cos = R.cosine_rows(phi, phi[base])
        worst = members[np.argsort(true_cos[members])[:F]]
        of = np.zeros(n, bool); of[worst] = True
        r2 = kmedoids.fastpam1(D, B, max_iter=100, init="build", random_state=0)
        lead_r, m_r = rolc_for_budget(leaves, scores, B)
        leads = {"cluster_only": base, "check": np.where(flagged, idx, base), "random_flag": np.where(rf, idx, base),
                 "oracle_flag": np.where(of, idx, base),
                 "more_medoids": np.asarray(r2.medoids).astype(np.int64)[np.asarray(r2.labels).astype(np.int64)],
                 "rolc": lead_r}
        for k, lead in leads.items():
            v[k] = stats(phi, lead)
            c = R.cosine_rows(phi, corrected(phi, leaves, lead, tab))
            v[k + "_corr"] = dict(expl=int(len(np.unique(lead))), cos=float(c.mean()), b07=float((c < 0.7).mean()),
                                  p10=float(np.percentile(c, 10)))
        for k, s in v.items():
            rows.append(dict(subject=name, build=int(b), n=n, k0=k0, flagged=F, budget=B, tau=tau, variant=k, **s))
        print(f"  {name} b={b} n={n} k0={k0} flagged={F} B={B}", flush=True)
    json.dump(dict(subject=name, tau=tau, frac=frac, rows=rows), open(os.path.join(out, name + ".json"), "w"))
    print(f"done {name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ddir, out, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for s in sys.argv[4:]:
        try:
            run_subject(ddir, s, out, frac)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", s, repr(e), flush=True)
