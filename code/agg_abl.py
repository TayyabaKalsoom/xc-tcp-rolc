import json, glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
for tag,d in (("k0 = n/8","abl_res_125"),("k0 = n/4","abl_res_25")):
    R=[]
    for f in glob.glob(f"{d}/*.json"): R+=json.load(open(f))["rows"]
    df=pd.DataFrame(R)
    subj=df.groupby(["subject","variant"])[["expl","cos","b07","p10"]].mean().reset_index()
    nsub=subj.subject.nunique(); nb=df.groupby("subject").build.nunique().sum()
    print(f"\n===== {tag}: {nsub} subjects, {nb} builds =====")
    med=subj.groupby("variant")[["expl","cos","b07","p10"]].median().round(3)
    order=["individual","cluster_only","check","random_flag","more_medoids","oracle_flag","rolc"]
    print(med.loc[order])
    piv=subj.pivot(index="subject",columns="variant",values="cos")
    flg=df.groupby("subject").flagged.mean(); bud=df.groupby("subject").budget.mean(); kk=df.groupby("subject").k0.mean()
    print("median k0, flagged, budget per subject:", kk.median(), flg.median(), bud.median())
    res=[]
    for a,b in [("check","random_flag"),("check","more_medoids"),("check","cluster_only"),("oracle_flag","check"),("rolc","check")]:
        diff=(piv[a]-piv[b]).dropna()
        try: p=wilcoxon(diff).pvalue
        except Exception: p=np.nan
        res.append((a,b,diff.median(),int((diff>0).sum()),int((diff<0).sum()),len(diff),p))
    pm=pd.DataFrame(res,columns=["A","B","median_cos_diff(A-B)","A_wins","B_wins","n","wilcoxon_p"]).round(4)
    print(pm.to_string(index=False))
    # below-0.7 share comparison
    pb=subj.pivot(index="subject",columns="variant",values="b07")
    print("share below 0.7 (median over subjects, %):", (pb[order[1:]].median()*100).round(2).to_dict())
    subj.to_csv(f"ablation_subject_{d}.csv",index=False)
