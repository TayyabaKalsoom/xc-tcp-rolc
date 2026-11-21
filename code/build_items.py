"""Build the study items from real TCP-CI builds.
Usage: python build_items.py <tcpci_dir> <out_dir> [subject ...]
Writes items_public.json (shown to participants, NO ground truth) and items_truth.csv (analysis only)."""
import sys, os, json, re, warnings
import numpy as np, pandas as pd, lightgbm as lgb, shap
warnings.filterwarnings("ignore"); os.environ["OMP_NUM_THREADS"] = "1"
import run_tcpci as R, improve_corrected as IC

SUBJECTS = ["Angel-ML@angel", "jcabi@jcabi-github", "Graylog2@graylog2-server", "apache@curator", "yamcs@Yamcs", "CompEvol@beast2"]
SEED = 2026

PREFIX = {"TES_": "test file", "REC_": "recent history of this test", "COV_": "coverage of changed/impacted code",
          "COD_COV_": "code covered by this test", "DET_": ""}
WORDS = {"COM": "code metrics", "PRO": "process metrics", "CHN": "change metrics", "C": "of covered and changed code", "IMP": "of covered and impacted code"}
METRIC = {"CountLine": "number of lines", "CountLineCode": "lines of code", "CountLineBlank": "blank lines", "CountLineComment": "comment lines",
          "CountStmt": "statements", "CountStmtDecl": "declaration statements", "CountStmtExe": "executable statements",
          "CountDeclFunction": "functions declared", "CountDeclClass": "classes declared", "CountDeclMethod": "methods declared",
          "MaxCyclomatic": "max cyclomatic complexity", "SumCyclomatic": "total cyclomatic complexity", "MaxNesting": "max nesting depth",
          "RatioCommentToCode": "comment-to-code ratio", "CommitCount": "number of commits", "DistinctDevCount": "number of distinct developers",
          "OwnersContribution": "owner's share of contribution", "OwnersExperience": "owner's experience", "AllCommitersExperience": "all committers' experience",
          "MinorContributorCount": "minor contributors", "LinesAdded": "lines added", "LinesDeleted": "lines deleted",
          "AddedChangeScattering": "scattering of added lines", "DeletedChangeScattering": "scattering of deleted lines",
          "DMMSize": "design-quality (size) of the change", "DMMComplexity": "design-quality (complexity) of the change", "DMMInterfacing": "design-quality (interfacing) of the change",
          "Age": "builds since first seen", "LastFailureAge": "builds since last failure", "LastTransitionAge": "builds since last pass/fail change",
          "RecentAvgExeTime": "recent average run time", "RecentMaxExeTime": "recent maximum run time", "RecentFailRate": "recent failure rate",
          "RecentAssertRate": "recent assertion-failure rate", "RecentExcRate": "recent exception rate", "RecentTransitionRate": "recent pass/fail change rate",
          "TotalAvgExeTime": "overall average run time", "TotalMaxExeTime": "overall maximum run time", "TotalFailRate": "overall failure rate",
          "TotalAssertRate": "overall assertion-failure rate", "TotalExcRate": "overall exception rate", "TotalTransitionRate": "overall pass/fail change rate",
          "LastVerdict": "result of the previous run", "LastExeTime": "run time of the previous run",
          "MaxTestFileFailRate": "highest failure rate in the test file", "MaxTestFileTransitionRate": "highest pass/fail change rate in the test file",
          "ChnScoreSum": "sum of change-coverage scores", "ImpScoreSum": "sum of impact-coverage scores", "ChnCount": "number of changed entities covered", "ImpCount": "number of impacted entities covered"}

def describe(name):
    for p in ("COD_COV_", "TES_", "REC_", "COV_"):
        if name.startswith(p):
            rest = name[len(p):]; break
    else: return name
    toks = rest.split("_")
    if p == "COD_COV_":
        grp, scope, metric = toks[0], toks[1], "_".join(toks[2:])
        m = METRIC.get(metric, re.sub(r"(?<!^)(?=[A-Z])", " ", metric).lower())
        return f"{m} ({WORDS.get(grp, grp)}, {WORDS.get(scope, scope)})"
    if p == "TES_":
        grp, metric = toks[0], "_".join(toks[1:])
        m = METRIC.get(metric, re.sub(r"(?<!^)(?=[A-Z])", " ", metric).lower())
        return f"{m} (test file, {WORDS.get(grp, grp)})"
    m = METRIC.get(rest, re.sub(r"(?<!^)(?=[A-Z])", " ", rest).lower())
    return f"{m} ({PREFIX[p]})"

def fmt(v):
    if abs(v) >= 1000 or float(v).is_integer(): return f"{v:,.0f}"
    return f"{v:.3g}"

def pct_text(col, v):
    if col.max() == col.min(): return "same for every test in this build"
    p = (col <= v).mean() * 100
    return f"percentile {p:.0f} in this build (100 = highest)"

def prep(ddir, name):
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
    model = lgb.LGBMRanker(n_estimators=R.TREES, num_leaves=R.LEAVES, learning_rate=0.1, objective="lambdarank", random_state=0, n_jobs=1, verbose=-1)
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
    trs = tr.iloc[np.linspace(0, len(tr) - 1, min(1000, len(tr))).astype(int)]
    tab, _ = IC.build_table(model, trs[feats].to_numpy(np.float32), F)
    cand = [b for b in ev_pool if R.MIN_TESTS <= len(grp.get_group(b)) and grp.get_group(b)["_fail"].sum() >= 1]
    cand = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, 2).astype(int))]
    return df, grp, feats, model, expl, tab, tau, cand

def top(v, k): return np.argsort(-np.abs(v), kind="stable")[:k]

def main():
    ddir, out = sys.argv[1], sys.argv[2]; subs = sys.argv[3:] or SUBJECTS
    os.makedirs(out, exist_ok=True); rng = np.random.default_rng(SEED)
    pool, practice = [], []
    for si, name in enumerate(subs, 1):
        df, grp, feats, model, expl, tab, tau, cand = prep(ddir, name)
        rows = []
        for b in cand:
            g = grp.get_group(b); X = g[feats].to_numpy(np.float32); n = len(g)
            leaves, scores = model.predict(X, pred_leaf=True), model.predict(X)
            phi = expl.shap_values(X); lead, isl = R.rolc_assign(leaves, scores, tau)
            corr = IC.corrected(phi, leaves, lead, tab)
            rank = (-scores).argsort(kind="stable").argsort() + 1
            gsize = np.bincount(lead, minlength=n)
            glob_imp = np.abs(phi).mean(0)
            for i in np.flatnonzero(~isl):
                rows.append(dict(subject=name, sid=f"P{si}", build=int(b), test=g["Test"].iloc[i], i=i, n=n, rank=int(rank[i]), score=float(scores[i]),
                                 cos=float(R.cosine_rows(phi[i:i+1], corr[i:i+1])[0]), cos_raw=float(R.cosine_rows(phi[i:i+1], phi[lead[i]:lead[i]+1])[0]),
                                 lead_i=int(lead[i]), gsize=int(gsize[lead[i]]), leader_rank=int(rank[lead[i]]),
                                 X=X, phi=phi, corr=corr, glob=glob_imp, fail=int(g["_fail"].iloc[i]), feats=feats))
        d = pd.DataFrame([{k: v for k, v in r.items() if k not in ("X", "phi", "corr", "glob", "feats")} for r in rows])
        q = d["cos"].quantile([1/3, 2/3]).to_numpy(); d["stratum"] = np.where(d["cos"] <= q[0], 0, np.where(d["cos"] <= q[1], 1, 2))
        chosen = []
        for s in (0, 1, 2):
            ids = d.index[d.stratum == s].to_numpy(); chosen.append(rng.choice(ids, size=min(2, len(ids)), replace=False).tolist())
        # practice: one extra mid-stratum not chosen
        used = {x for c in chosen for x in c}; extra = [x for x in d.index[d.stratum == 1] if x not in used]
        if si == 1: practice = [rows[x] for x in rng.choice(extra, 2, replace=False)]
        # set assignment: A:(low,mid) B:(mid,high) C:(high,low)  -> each set balanced across strata over 6 subjects
        asg = {"A": [chosen[0][0], chosen[1][0]], "B": [chosen[1][1], chosen[2][0]], "C": [chosen[2][1], chosen[0][1]]}
        for sset, ids in asg.items():
            for x in ids: pool.append((sset, rows[x], int(d.loc[x, "stratum"])))
        print(name, "tau", tau, "items", sum(len(v) for v in asg.values()), "strata cos", d.groupby("stratum")["cos"].mean().round(3).to_dict(), flush=True)
    def make(sset, r, stratum, idx):
        i = r["i"]; feats = r["feats"]; phi_own, phi_grp = r["phi"][i], r["corr"][i]
        own3, grp3 = top(phi_own, 3), top(phi_grp, 3)
        cand = list(dict.fromkeys(list(own3) + list(grp3)))
        for j in np.argsort(-r["glob"], kind="stable"):
            if len(cand) >= 8: break
            if j not in cand: cand.append(int(j))
        cand = [int(c) for c in cand[:8]]; rng.shuffle(cand)
        col = {j: r["X"][:, j] for j in cand}
        fe = [dict(key=feats[j], desc=describe(feats[j]), value=fmt(float(r["X"][i, j])), median=fmt(float(np.median(col[j]))), pct=pct_text(col[j], r["X"][i, j])) for j in cand]
        ex = lambda v: [dict(key=feats[j], desc=describe(feats[j]), contrib=round(float(v[j]), 4)) for j in top(v, 5)]
        pub = dict(id=idx, set=sset, project=r["sid"], test=f"T-{rng.integers(100, 999)}", rank=r["rank"], n=r["n"],
                   features=fe, expl_own=ex(phi_own), expl_group=ex(phi_grp), group_size=r["gsize"], leader_rank=r["leader_rank"])
        tru = dict(item_id=idx, set=sset, subject=r["subject"], project=r["sid"], build=r["build"], test_id=r["test"], n=r["n"], rank=r["rank"], failed=r["fail"],
                   stratum=stratum, cosine_group_vs_own=round(r["cos"], 4), cosine_uncorrected=round(r["cos_raw"], 4), group_size=r["gsize"],
                   truth_top3="|".join(feats[j] for j in own3), group_top3="|".join(feats[j] for j in grp3),
                   copy_accuracy=len(set(own3) & set(grp3)) / 3, candidates="|".join(feats[j] for j in cand))
        return pub, tru
    pubs, trus = [], []
    for k, (sset, r, st) in enumerate(pool, 1):
        p, t = make(sset, r, st, f"{sset}{k:02d}"); pubs.append(p); trus.append(t)
    # renumber ids within set
    cnt = {}
    for p, t in zip(pubs, trus):
        cnt[p["set"]] = cnt.get(p["set"], 0) + 1; new = f'{p["set"]}{cnt[p["set"]]:02d}'; p["id"] = new; t["item_id"] = new
    prac = []
    for k, r in enumerate(practice, 1):
        p, t = make("PRACTICE", r, 1, f"X{k}"); p["truth_top3"] = [dict(key=x, desc=describe(x)) for x in t["truth_top3"].split("|")]; prac.append(p)
    json.dump(dict(items=pubs, practice=prac), open(os.path.join(out, "items_public.json"), "w"))
    pd.DataFrame(trus).to_csv(os.path.join(out, "items_truth.csv"), index=False)
    print("items", len(pubs), "practice", len(prac))

if __name__ == "__main__": main()
