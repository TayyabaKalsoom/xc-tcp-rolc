# ROLC
Code, result files and converted data for the paper on rank-ordered leader coverage (ROLC) for explaining LambdaMART test-case rankings. Every table in the paper can be traced to a CSV in `results/aggregated/`.


## What is in each folder
| Folder | Content |
|---|---|
| `code/` | All scripts that produced the numbers: `run_tcpci.py` (main comparison), `e2e.py` (end-to-end time/memory), `ablation.py` and `ablation_corr.py` (equal-budget ablation, without / with correction), `improve_corrected.py`, `improve_fair.py`, `improve_ik25.py` (leaf-table correction; interventional TreeSHAP and KernelSHAP), `convert_rtp.py` (RTPTorrent to TCP-CI schema), `agg*.py` (aggregation), `get_rtp.py` |
| `results/aggregated/` | The CSVs behind every table: `tcpci25_subject_summary.csv`, `tcpci25_build_level.csv`, `improve25_subject.csv`, `improve25_build.csv`, `e2e25_*.csv`, `abl25_*.csv`, `ik25_*.csv`, `ablation_corr_abc*.csv`, and `rtp/` for RTPTorrent. `sandbox17_*` are the earlier 17-subject sandbox files, kept for reference only |
| `results/raw_user_25_subject_runs/` | Per-subject raw JSON outputs of the 25-subject runs (`res`, `e2e_res`, `abl_res_125`, `abl_res_25`, `imp_res`, `ik_res`, `ik_res_kernel`) |
| `results/raw_sandbox_runs/` | Raw outputs of the runs made in the Claude workspace (RTPTorrent, corrected ablation `abc1`/`abc2`, fairness check) |
| `datasets/` | `rtptorrent_converted/` (16 projects in TCP-CI schema), `tcpci_samples/` (3 small TCP-CI subjects), `tcpci25_subject_manifest.csv` |
| `figures/` | The three figures used in the paper (PNG) |

## Datasets not bundled
The full TCP-CI benchmark (25 subjects, about 1.7 GB unpacked) is public: https://zenodo.org/records/5532640. Unzip it so that each subject is `<datasets-dir>/<owner>@<repo>/dataset.csv`. RTPTorrent raw data is public (Mattis et al., MSR 2020); `datasets/rtptorrent_converted/` is the already converted version.

## How to reproduce (Windows example; set the dataset folder first)
```
pip install lightgbm shap kmedoids scikit-learn pandas numpy scipy
python code\run_tcpci.py D:\xc\datasets D:\xc\res  <subject1> <subject2> ...
python code\e2e.py       D:\xc\datasets D:\xc\e2e_res <subjects...>
python code\ablation.py  D:\xc\datasets D:\xc\abl_res_125 0.125 <subjects...>   # k0 = n/8 (use 0.25 for n/4)
python code\ablation_corr.py D:\xc\datasets D:\xc\abc1 0.125 <subjects...>      # ablation with correction
python code\improve_corrected.py D:\xc\datasets D:\xc\imp_res 1500 <subjects...> # leaf-table correction
python code\improve_ik25.py D:\xc\datasets D:\xc\ik_res <subjects...>          # set KERNEL=0 to skip KernelSHAP
python code\convert_rtp.py <rtptorrent-dir> <out-dir>
```

6. **RQ5 / Table 8** (fidelity for top-ranked and failing tests) uses all 25 subjects: `code/fidelity_where_it_matters.py` (needs `build_items.py`, `improve_corrected.py`, `run_tcpci.py` in the same folder), per-subject results in `results/aggregated/fidelity_where_it_matters_25.csv` and raw JSON in `results/raw_user_25_subject_runs/fw_res/`. `fidelity_where_it_matters_sandbox16.csv` is the earlier 16-subject check.
7. Per-subject cosines are medians over subjects of per-subject means; the median across subjects is 0.921 (ROLC) and 0.987 (corrected).
