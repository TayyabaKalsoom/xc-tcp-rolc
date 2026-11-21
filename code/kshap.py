import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/claude/work")
import shap
from rolc_smoke import make_data, train_ranker
model = train_ranker()
Xb,_ = make_data(50, seed=5)
Xt,_ = make_data(10, seed=6)
f = lambda x: model.predict(x)
ke = shap.KernelExplainer(f, Xb)
t=time.perf_counter(); ke.shap_values(Xt, nsamples=500, silent=True); dt=(time.perf_counter()-t)/len(Xt)
print(f"KernelSHAP per test case (nsamples=500): {dt:.3f} s")
for n,m in [(500,170),(1000,250),(2000,429),(4000,637),(8000,897)]:
    print(n, f"individual {dt*n:.0f}s  ROLC-explained {dt*m:.0f}s  ratio {m/n:.2f}")
