"""
Real-data experiment: explanation reuse methods on TCP-CI subjects.

Per subject (chronological split by build id):
  - first 55% of builds -> train LightGBM LambdaMART (binary label Verdict>0)
  - next ~5% of builds  -> calibration pool for the ROLC threshold tau
  - last 40% of builds  -> evaluation; up to NB evenly spaced builds that have
                            >=1 failing test and >=MIN_TESTS tests
Features: every column except Build, Test, Verdict, Duration (execution outcome
of the same build) and the DET_* columns (fault-detection counters).

Methods, all explaining the same number m of test cases (m set by ROLC):
  ROLC, K-Medoids (FastPAM), K-Means (sparse one-hot leaves), CLARA-style K-Medoids.
Fidelity = cosine(own TreeSHAP vector, vector of the test case whose explanation is reused).
Ranking is every test case's own LambdaMART score for all methods, so APFD is
identical across methods and is reported once.

usage: python3 run_tcpi.py <datasets_dir> <out_dir> <subject1> [subject2 ...]
"""
import sys, os, time, json, warnings, tracemalloc
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, scipy.sparse as sp
import lightgbm as lgb, shap, kmedoids
from sklearn.cluster import KMeans

TRAIN_FRAC, CAL_FRAC = 0.55, 0.05
MIN_TESTS, TARGET_COS = 30, 0.90
TREES, LEAVES = 100, 15


# ---------------- methods ----------------
def cosine_rows(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def rolc_assign(leaves, scores, tau, refine=True):
    n, T = leaves.shape
    order = np.argsort(-scores, kind="stable")
    leader_of = np.full(n, -1, dtype=np.int64)
    is_leader = np.zeros(n, dtype=bool)
    L = np.empty((n, T), dtype=leaves.dtype)
    ids = np.empty(n, dtype=np.int64)
    m = 0
    for i in order:
        if m:
            agree = (L[:m] == leaves[i]).mean(axis=1)
            j = int(agree.argmax())
            if agree[j] >= tau:
                leader_of[i] = ids[j]
                continue
        L[m] = leaves[i]; ids[m] = i; leader_of[i] = i; is_leader[i] = True; m += 1
    if refine and m > 1:
        Lm = L[:m]
        for i in np.flatnonzero(~is_leader):
            leader_of[i] = ids[int((Lm == leaves[i]).mean(axis=1).argmax())]
    return leader_of, is_leader


def leaf_dist(leaves, block=256):
    n = len(leaves)
    D = np.empty((n, n), dtype=np.float32)
    for s in range(0, n, block):
        e = min(n, s + block)
        D[s:e] = 1.0 - (leaves[s:e, None, :] == leaves[None, :, :]).mean(2)
    return D


def kmedoids_assign(leaves, k):
    D = leaf_dist(leaves)
    r = kmedoids.fastpam1(D, k, max_iter=100, init="build", random_state=0)
    return np.asarray(r.medoids).astype(np.int64)[np.asarray(r.labels).astype(np.int64)]


def onehot(leaves):
    n, T = leaves.shape
    cols = (np.arange(T)[None, :] * LEAVES + leaves).ravel()
    rows = np.repeat(np.arange(n), T)
    return sp.csr_matrix((np.ones(n * T, np.float32), (rows, cols)), shape=(n, T * LEAVES))


def kmeans_assign(leaves, k):
    Xs = onehot(leaves)
    mdl = KMeans(k, n_init=1, max_iter=50, random_state=0).fit(Xs)
    lab, C = mdl.labels_, mdl.cluster_centers_
    leader = np.empty(len(leaves), np.int64)
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        leader[idx] = idx[((Xs[idx].toarray() - C[c]) ** 2).sum(1).argmin()]
    return leader


def clara_assign(leaves, k, seed=0):
    rng = np.random.default_rng(seed); n = len(leaves)
    s = min(n, max(400, 2 * k))
    samp = rng.choice(n, s, replace=False)
    med = np.asarray(kmedoids.fastpam1(leaf_dist(leaves[samp]), k, max_iter=50, init="build").medoids).astype(np.int64)
    meds = samp[med]; Lm = leaves[meds]
    leader = np.empty(n, np.int64)
    for i in range(n):
        leader[i] = meds[(Lm == leaves[i]).mean(1).argmax()]
    return leader


def timed(f):
    t = time.perf_counter(); r = f(); return r, time.perf_counter() - t


def peak_mb(f):
    tracemalloc.start(); f(); _, p = tracemalloc.get_traced_memory(); tracemalloc.stop(); return p / 1e6


def apfd(scores, fail):
    n = len(scores); m = int(fail.sum())
    if m == 0 or n < 2: return np.nan
    order = np.argsort(-scores, kind="stable")
    pos = np.flatnonzero(fail[order]) + 1
    return 1 - pos.sum() / (m * n) + 1 / (2 * n)


# ---------------- per-subject run ----------------
def run_subject(ddir, name, out):
    t0 = time.time()
    df = pd.read_csv(os.path.join(ddir, name, "dataset.csv"))
    drop = [c for c in df.columns if c in ("Build", "Test", "Verdict", "Duration") or c.startswith("DET_")]
    feats = [c for c in df.columns if c not in drop]
    df["_fail"] = (df["Verdict"] > 0).astype(int)
    builds = np.sort(df["Build"].unique())
    nb = len(builds)
    tr_b = builds[: int(nb * TRAIN_FRAC)]
    cal_b = builds[int(nb * TRAIN_FRAC): int(nb * (TRAIN_FRAC + CAL_FRAC)) + 1]
    ev_pool = builds[int(nb * (TRAIN_FRAC + CAL_FRAC)):]
    df = df.sort_values(["Build", "Test"], kind="stable")
    grp = df.groupby("Build", sort=True)

    tr = df[df["Build"].isin(tr_b)]
    Xtr = tr[feats].to_numpy(np.float32); ytr = tr["_fail"].to_numpy()
    gsz = tr.groupby("Build", sort=True).size().to_numpy()
    model = lgb.LGBMRanker(n_estimators=TREES, num_leaves=LEAVES, learning_rate=0.1, objective="lambdarank",
                           random_state=0, n_jobs=1, verbose=-1)
    model.fit(Xtr, ytr, group=gsz)
    expl = shap.TreeExplainer(model)

    # calibration pool (held out of training)
    cal = df[df["Build"].isin(cal_b)]
    cal = cal[cal.groupby("Build")["Build"].transform("size") >= MIN_TESTS]
    if len(cal) > 3000: cal = cal.iloc[:3000]
    if len(cal) < 50:
        cal = tr.groupby("Build").filter(lambda g: len(g) >= MIN_TESTS).tail(1500)
    Xc = cal[feats].to_numpy(np.float32)
    lc, sc, pc = model.predict(Xc, pred_leaf=True), model.predict(Xc), expl.shap_values(Xc)
    tau, ccal = None, None
    for t in np.linspace(0.3, 0.95, 14):
        lo, _ = rolc_assign(lc, sc, t)
        c = cosine_rows(pc, pc[lo]).mean()
        tau, ccal = float(t), float(c)
        if c >= TARGET_COS: break

    # evaluation builds
    cand = []
    for b in ev_pool:
        g = grp.get_group(b)
        if len(g) >= MIN_TESTS and g["_fail"].sum() >= 1:
            cand.append((b, len(g)))
    max_b = 3 if name.startswith("JMRI") else 8
    if len(cand) > max_b:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, max_b).astype(int)]
    big = max(cand, key=lambda x: x[1])[0] if cand else None

    rows = []
    for b, n in cand:
        g = grp.get_group(b)
        X = g[feats].to_numpy(np.float32); fail = g["_fail"].to_numpy()
        leaves, scores = model.predict(X, pred_leaf=True), model.predict(X)
        phi, t_ind = timed(lambda: expl.shap_values(X))
        (lo, isl), t_rolc = timed(lambda: rolc_assign(leaves, scores, tau))
        m = int(isl.sum())
        if m >= n: continue
        _, t_expl_m = timed(lambda: expl.shap_values(X[isl]))
        r = dict(subject=name, build=int(b), n=n, failing=int(fail.sum()), m=m, compress=n / m, tau=tau,
                 apfd=float(apfd(scores, fail)), t_ind=t_ind, t_expl_m=t_expl_m, train_calib_cos=ccal)
        methods = {"ROLC": lambda: rolc_assign(leaves, scores, tau)[0],
                   "KMedoids": lambda: kmedoids_assign(leaves, m),
                   "KMeans": lambda: kmeans_assign(leaves, m),
                   "CLARA": lambda: clara_assign(leaves, m)}
        for mn, f in methods.items():
            lead, dt = timed(f)
            c = cosine_rows(phi, phi[lead])
            r[f"{mn}_t_assign"] = dt
            r[f"{mn}_cos"] = float(c.mean())
            r[f"{mn}_below07"] = float((c < 0.7).mean())
            r[f"{mn}_explained"] = int(len(set(lead.tolist())))
            if b == big:
                r[f"{mn}_mem_mb"] = peak_mb(f)
        rows.append(r)
        print(f"  {name} build {b} n={n} m={m}", flush=True)
    json.dump(dict(subject=name, tau=tau, calib_cos=ccal, n_builds=int(nb), n_features=len(feats),
                   n_train_rows=int(len(tr)), eval_builds=len(rows), rows=rows, secs=time.time() - t0),
              open(os.path.join(out, name + ".json"), "w"), indent=1)
    print(f"done {name} in {time.time()-t0:.0f}s ({len(rows)} eval builds, tau={tau:.2f})", flush=True)


if __name__ == "__main__":
    ddir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for s in sys.argv[3:]:
        try:
            run_subject(ddir, s, out)
        except Exception as e:
            print("FAILED", s, repr(e), flush=True)
