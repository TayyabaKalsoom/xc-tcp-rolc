import json, glob, numpy as np, pandas as pd
R=[]
for f in sorted(glob.glob("e2e_res/*.json")): R+=json.load(open(f))["rows"]
df=pd.DataFrame(R)
print("subjects",df.subject.nunique(),"builds",len(df))
M=["ROLC","CLARA","KMedoids"]
rows=[]
for ex,nm in (("A","TreeSHAP (path-dependent)"),("B","TreeSHAP (interventional, 100 bg)"),("C","KernelSHAP")):
    for m in M:
        df[f"{ex}_speed_{m}"]=df[f"{ex}_total_Individual"]/df[f"{ex}_total_{m}"]
    g=df.groupby("subject")
    sub=pd.DataFrame({"share":g[f"{ex}_expl_share_Individual"].median()})
    for m in M: sub[m]=g[f"{ex}_speed_{m}"].median()
    print(f"\n=== Explainer {ex}: {nm} ===  (median over subjects; speedup = individual pipeline time / reuse pipeline time)")
    print("explanation share of individual pipeline (median):", round(sub.share.median(),3))
    for m in M:
        s=sub[m].dropna(); print(f"  {m:9s} speedup median {s.median():.3f}x  (saving {100*(1-1/s.median()):.1f}%)  range {s.min():.2f}-{s.max():.2f}  n_subj={len(s)}")
    sub.round(3).to_csv(f"e2e_subject_{ex}.csv")
print("\nmedian train time per build (s):", df.t_train.median().round(3), " explanation A (indiv) median ms:", (df.expA_ind.median()*1000).round(1))
print("compression n/m median:", (df.n/df.m).median().round(2))
mem=df.dropna(subset=["mem_Individual_mb"])
mm=mem.groupby("subject")[[c for c in mem.columns if c.startswith("mem_")]].max()
mm["n"]=mem.groupby("subject").n.max()
mm["out_ratio"]=mem.groupby("subject").out_bytes_reuse.max()/mem.groupby("subject").out_bytes_Individual.max()
pd.set_option("display.width",200)
print("\n=== peak memory of explanation phase (tracemalloc, MB; largest eval build per subject) ===")
print(mm.round(2))
print("median ratio ROLC/Individual:", (mm.mem_ROLC_mb/mm.mem_Individual_mb).median().round(3),
      "| CLARA/Ind:", (mm.mem_CLARA_mb/mm.mem_Individual_mb).median().round(3),
      "| KMedoids/Ind:", (mm.mem_KMedoids_mb/mm.mem_Individual_mb).dropna().median().round(3))
print("analytic output storage reuse/individual median:", mm.out_ratio.median().round(3))
df.to_csv("e2e_build_level.csv",index=False); mm.round(3).to_csv("e2e_memory.csv")
