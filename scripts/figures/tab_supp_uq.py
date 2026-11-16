"""Supplementary Tables S14/S15: PICP and MPIW of all four parameters at 95% and 68%,
every noise condition, five interval methods. Source: n1_u1_rn_noprof.json (3 seeds, main network; 11-16 Option 1)."""
import json, sys
import os
J = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DUALTCN_RUNS", "/work/hdd/bgte/kahmed2/DualTCN_runs") + "/analysis/n1_u1/n1_u1_rn_noprof.json"   # 11-16 Option 1: main network (contains conformal_trainmix)
d = json.load(open(J))["conditions"]
# D5 (11-15): training-noise calibration from the rerun with the same seeds (n1_u1_rn_base_tm.json; all shared
# entries agree with n1_u1_rn_base.json to <= 0.005)
_tm = J.replace("n1_u1_rn_base.json", "n1_u1_rn_base_tm.json")
TM = json.load(open(_tm if os.path.exists(_tm) else J))["conditions"]   # a current run of n1_u1_noise_uq.py contains both
for _c in d:
    for _l in ("0.95", "0.68"):
        d[_c]["uq"][_l]["conformal_trainmix"] = TM[_c]["uq"][_l]["conformal_trainmix"]
CONDS = [("clean", "Clean"), ("snr40", "40\\,dB"), ("snr30", "30\\,dB"), ("snr20", "20\\,dB"),
         ("snr10", "10\\,dB$^\\dagger$"), ("gain1", "Gain 1\\%"), ("gain2", "Gain 2\\%"),
         ("gain5", "Gain 5\\%"), ("snr30_gain2", "30\\,dB + 2\\%")]
METH = [("mcdropout_gauss", "MC dropout"), ("ensemble_gauss", "Seed spread"),
        ("conformal_clean", "Conf. clean"), ("conformal_norm_clean", "Conf. norm. clean"),
        ("conformal_trainmix", "Conf. train. noise"),
        ("conformal_same_condition", "Conf. matched$^\\ast$")]
R = "\\color{red}"
def f(x): return f"{x:.2f}"
out = []
GROUPS = [("snr", "for clean data and white noise", CONDS[:5]), ("gain", "for receiver gain errors and combined noise", CONDS[5:])]
for lev, lab, pw in (("0.95", "95", "0.95 for $\\sigma_1$ and $\\sigma_2$ and 0.94 and 0.93 for $d_1$ and $d_2$"),
                     ("0.68", "68", "0.68 for $\\sigma_1$ and $\\sigma_2$ and 0.65 and 0.61 for $d_1$ and $d_2$")):
    for gk, gtxt, conds in GROUPS:
        out.append("\\begin{table*}[p]\n\\centering")
        if gk == "snr":
            cap = (f"\\caption{{\\rev{{Coverage (PICP) and mean width (MPIW, normalized units) of the nominal {lab}\\% prediction intervals for all four parameters, {gtxt} (main network, 150,000 test samples). MC dropout: 20 passes; Seed spread: Gaussian interval from the mean and standard deviation of the three seeds; Conf. clean: split conformal on the three-seed mean, calibrated on 50,000 clean validation samples; Conf. norm. clean: the same, with residuals scaled by the seed standard deviation; Conf. train. noise: calibrated on validation samples with the noise distribution of the training data, which does not require the noise level of the survey; Conf. matched: calibrated on validation samples with the same noise condition as the test data, which requires the noise level of the survey to be known ($^\\ast$upper reference). Widths are not clipped to $[0,1]$; the central {lab}\\% interval of the prior has a width of {pw}. $^\\dagger$Outside the SNR range of the training data.}}}}")
        else:
            cap = f"\\caption{{\\rev{{As Table~\\ref{{tab:supp_uq{lab}}}, {gtxt}.}}}}"
        out.append(cap)
        out.append(f"\\label{{tab:supp_uq{lab}{'' if gk == 'snr' else 'g'}}}\n\\scriptsize\n\\setlength{{\\tabcolsep}}{{3pt}}")
        out.append("\\begin{tabular}{llcccccccc}\n\\toprule")
        out.append(" & & \\multicolumn{4}{c}{PICP} & \\multicolumn{4}{c}{MPIW} \\\\")
        out.append("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
        out.append("Condition & Method & $\\sigma_1$ & $\\sigma_2$ & $d_1$ & $d_2$ & $\\sigma_1$ & $\\sigma_2$ & $d_1$ & $d_2$ \\\\")
        for ck, cl in conds:
            out.append("\\midrule")
            for j, (mk, ml) in enumerate(METH):
                u = d[ck]["uq"][lev][mk]
                cells = [R + (cl if j == 0 else ""), R + ml] + [R + f(v) for v in u["picp"]] + [R + f(v) for v in u["mpiw"]]
                out.append(" & ".join(cells) + " \\\\")
        out.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
print("\n".join(out))
