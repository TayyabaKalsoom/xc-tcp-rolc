import json, glob, numpy as np, pandas as pd
R=[]; S=[]
for f in sorted(glob.glob("/tmp/u25/res/*.json")):
    d=json.load(open(f)); S.append(dict(subject=d["subject"],tau=d["tau"],calib_cos=d["calib_cos"],builds=d["n_builds"],feats=d["n_features"],train_rows=d["n_train_rows"]))
    R+=d["rows"]
df=pd.DataFrame(R); S=pd.DataFrame(S).set_index("subject")
M=["ROLC","KMedoids","KMeans","CLARA"]
for m in M: df[f"{m}_e2e"]=(df[f"{m}_t_assign"]+df["t_expl_m"])/df["t_ind"]
g=df.groupby("subject")
sub=pd.DataFrame({"n_med":g["n"].median(),"m_med":g["m"].median(),"compress":g["compress"].median(),"apfd":g["apfd"].median(),"tau":S["tau"],"calib_cos":S["calib_cos"]})
for m in M:
    sub[f"{m}_cos"]=g[f"{m}_cos"].mean(); sub[f"{m}_b07"]=g[f"{m}_below07"].mean()*100
    sub[f"{m}_t"]=g[f"{m}_t_assign"].median(); sub[f"{m}_e2e"]=g[f"{m}_e2e"].median(); sub[f"{m}_mem"]=g[f"{m}_mem_mb"].max()
sub["indiv_ms"]=g["t_ind"].median()*1000
pd.set_option("display.width",250); pd.set_option("display.max_columns",60)
print(sub.round(3)[["n_med","m_med","compress","apfd","tau","calib_cos"]])
print(sub.round(3)[[f"{m}_cos" for m in M]+[f"{m}_b07" for m in M]])
print(sub.round(4)[[f"{m}_t" for m in M]+[f"{m}_mem" for m in M]])
print(sub.round(3)[[f"{m}_e2e" for m in M]])
print("\n=== MEDIAN OVER 25 SUBJECTS ===")
med=sub.median(numeric_only=True); print(med.round(3).to_string())
print("\nsubjects with mean cosine >=0.90:", {m:int((sub[f'{m}_cos']>=0.90).sum()) for m in M})
print("subjects with mean cosine >=0.80:", {m:int((sub[f'{m}_cos']>=0.80).sum()) for m in M})
print("compress range", sub.compress.min().round(2), sub.compress.max().round(2), "| builds:", len(df))
print("APFD median", df.apfd.median().round(3))
sub.round(4).to_csv("agg25/tcpci25_subject_summary.csv"); df.to_csv("agg25/tcpci25_build_level.csv",index=False)
# win/loss KMeans vs KMedoids fidelity & ROLC vs KMedoids
for a,b in [("KMeans","KMedoids"),("ROLC","KMedoids"),("CLARA","KMedoids"),("ROLC","KMeans")]:
    print(a,"vs",b,"cos higher on",int((sub[f"{a}_cos"]>sub[f"{b}_cos"]).sum()),"of 25 subjects; median diff",round((sub[f"{a}_cos"]-sub[f"{b}_cos"]).median(),4))
big=sub.sort_values("n_med").tail(4)[["n_med"]+[f"{m}_t" for m in M]+[f"{m}_mem" for m in M]]; print(big.round(3))
