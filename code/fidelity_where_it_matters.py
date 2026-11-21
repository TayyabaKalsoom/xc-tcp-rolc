"""Fidelity of borrowed explanations where testers look: top-ranked tests, failing tests, and decision-relevant measures.
Usage: python fidelity_where_it_matters.py <tcpci_dir> <out_dir> <subject> [maxb]
Per subject writes <out_dir>/<subject>.json with mean metrics by segment for ROLC borrowed as is ('raw') and with the leaf-table correction ('corr').
Metrics: cos = cosine with own SHAP vector; top1 = same most influential feature; top3 = overlap of top-3 features (0-1);
sign3 = share of the true top-3 features whose direction (raises/lowers) is the same; rho = Spearman correlation of |contribution| across features.
Leaders are explained individually, so they score 1.0 on every metric. Segments: all tests; borrowers only; top-10 ranked tests; failing tests; failing borrowers."""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import run_tcpci as R, improve_corrected as IC, build_items as BI

def rank_rows(A): return np.argsort(np.argsort(A, axis=1, kind="stable"), axis=1).astype(float)
def metrics(phi, est):
    a = np.abs(phi); b = np.abs(est); n = len(phi)
    cos = R.cosine_rows(phi, est)
    t1 = (a.argmax(1) == b.argmax(1)).astype(float)
    ta = np.argsort(-a, axis=1)[:, :3]; tb = np.argsort(-b, axis=1)[:, :3]
    t3 = np.array([len(set(x) & set(y)) / 3 for x, y in zip(ta, tb)])
    s3 = np.array([(np.sign(est[i, ta[i]]) == np.sign(phi[i, ta[i]])).mean() for i in range(n)])
    ra, rb = rank_rows(a), rank_rows(b); ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    rho = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)
    return dict(cos=cos, top1=t1, top3=t3, sign3=s3, rho=rho)

def main():
    ddir, out, name = sys.argv[1], sys.argv[2], sys.argv[3]; maxb = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(out, exist_ok=True)
    df, grp, feats, model, expl, tab, tau, _ = BI.prep(ddir, name)
    builds = np.sort(df["Build"].unique()); nb = len(builds)
    pool = builds[int(nb * (R.TRAIN_FRAC + R.CAL_FRAC)):]
    cand = [b for b in pool if R.MIN_TESTS <= len(grp.get_group(b)) and grp.get_group(b)["_fail"].sum() >= 1]
    if len(cand) > maxb: cand = [cand[i] for i in np.linspace(0, len(cand) - 1, maxb).astype(int)]
    acc = {v: {m: [] for m in ["cos", "top1", "top3", "sign3", "rho"]} for v in ("raw", "corr")}
    seg = {m: [] for m in ["isl", "rank", "fail", "build"]}
    for b in cand:
        g = grp.get_group(b); X = g[feats].to_numpy(np.float32); n = len(g)
        leaves, scores = model.predict(X, pred_leaf=True), model.predict(X)
        phi = expl.shap_values(X); lead, isl = R.rolc_assign(leaves, scores, tau)
        corr = IC.corrected(phi, leaves, lead, tab); raw = phi[lead]
        rank = (-scores).argsort(kind="stable").argsort() + 1
        for v, est in (("raw", raw), ("corr", corr)):
            est = est.copy(); est[isl] = phi[isl]       # leaders are exact
            for k, arr in metrics(phi, est).items(): acc[v][k].append(arr)
        seg["isl"].append(isl); seg["rank"].append(rank); seg["fail"].append(g["_fail"].to_numpy().astype(bool)); seg["build"].append(np.full(n, b))
    cat = lambda L: np.concatenate(L)
    isl, rank, fail = cat(seg["isl"]), cat(seg["rank"]), cat(seg["fail"])
    segs = {"all": np.ones_like(isl, bool), "borrowers": ~isl, "top10": rank <= 10, "failing": fail, "failing_borrowers": fail & ~isl}
    res = dict(subject=name, builds=len(cand), tests=int(len(isl)), leader_share_all=float(isl.mean()), leader_share_top10=float(isl[rank <= 10].mean()),
               leader_share_failing=float(isl[fail].mean()), n_failing=int(fail.sum()), n_failing_borrowers=int((fail & ~isl).sum()), tau=tau)
    for v in ("raw", "corr"):
        for k in acc[v]:
            a = cat(acc[v][k])
            for s, mask in segs.items(): res[f"{v}_{k}_{s}"] = float(a[mask].mean()) if mask.any() else None
    json.dump(res, open(os.path.join(out, name + ".json"), "w"))
    print(name, "builds", len(cand), "leaders top10 %.2f failing %.2f all %.2f" % (res["leader_share_top10"], res["leader_share_failing"], res["leader_share_all"]),
          "| cos failing raw %.3f corr %.3f" % (res["raw_cos_failing"], res["corr_cos_failing"]), flush=True)
if __name__ == "__main__": main()
