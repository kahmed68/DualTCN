#!/usr/bin/env python3
"""O2: three-layer departure test (EMS revision, R2.1; queued by Khaled 2026-11-14).

Host models: the training prior (normalised log10 params uniform on [0,1]^4), two-layer seabed sigma2.
Departure: a thin resistive layer inside the seabed, at burial depth b ~ U(0, 50) m below the seafloor,
thickness h ~ logU(5, 50) m, conductivity sigma2 / c with contrast c ~ logU(10, 100).
Targets: sigma1, d1, d2 exact; sigma2 = host (background) conductivity.
Control: the SAME host models without the layer (paired), simulated with forward_torch.
Three-layer data: empymod.dipole with the settings that forward_torch reproduces (Key 201 2009 DLF, pts_per_dec 0,
xdirect False). GATE: for 200 host models (no layer, layer conductivity = sigma2), empymod vs forward_torch features
must agree (max |dE_norm| < 1e-8, max |dlogamp| < 1e-8), else abort.
Metrics per run: R2 per parameter on 3L and control, trace/log-amp RMSE of the re-simulated predictions (2-layer
operator), AUC of the per-sample misfit (3L vs control) and Spearman(misfit, parameter error) on 3L.
Also the data effect of the layer: median |delta log-amp| (decades) and median trace RMS change between 3L and control.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='rn_base_s1,rn_base_s2,rn_base_s3,rn_mis_s1')
ap.add_argument('--root', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/e1')
ap.add_argument('--arch', default='dualtcn')
ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--n-gate', type=int, default=200)
ap.add_argument('--seed', type=int, default=2101)
ap.add_argument('--batch', type=int, default=500)
ap.add_argument('--procs', type=int, default=16)
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/o2')
ap.add_argument('--tag', default='rn')
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import empymod
from config_v6 import RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT

OFF = np.asarray(RECEIVER_OFFSETS, float)


def em3(args):                                      # one model -> (R,F) complex
    s1, s2, d1, d2, b, h, s3 = args
    zsf = d1 + d2
    depth = [0.0, zsf] if (h <= 0) else ([0.0, zsf, zsf + b, zsf + b + h] if b > 0 else [0.0, zsf, zsf + h])
    res = [1e23, 1 / s1, 1 / s2] if h <= 0 else ([1e23, 1 / s1, 1 / s2, 1 / s3, 1 / s2] if b > 0 else [1e23, 1 / s1, 1 / s3, 1 / s2])
    E = empymod.dipole(src=[0, 0, d1], rec=[OFF, np.zeros_like(OFF), Z_OBS], depth=depth, res=res, freqtime=FREQS_V6,
                       ab=11, xdirect=False, ht='dlf', htarg={'dlf': 'key_201_2009', 'pts_per_dec': 0}, verb=0)
    return np.asarray(E).T                          # (R,F)


def main():
    import torch
    from scipy.stats import spearmanr
    import forward_torch as ft
    from train_v6 import ARCHS, NAMES, BOUNDS, misfit_terms, make_x
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # 11-16: TF32 off (A40 TF32 convs shift R2)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    LO, HI = BOUNDS[:, 0].double().numpy(), BOUNDS[:, 1].double().numpy()
    rng = np.random.default_rng(a.seed)
    th = rng.uniform(0, 1, (a.n, 4)); lp = LO + th * (HI - LO); p = 10 ** lp
    b = rng.uniform(0, 50, a.n); h = 10 ** rng.uniform(np.log10(5), np.log10(50), a.n)
    c = 10 ** rng.uniform(1, 2, a.n); s3 = p[:, 1] / c

    def feats_ft(pp):
        E, L = [], []
        with torch.no_grad():
            for i in range(0, len(pp), a.batch):
                q = torch.tensor(pp[i:i + a.batch], dtype=torch.float64, device=dev)
                En, la = ft.time_features(ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6), N_TIME)
                E.append(En.float()); L.append(la.float())
        return torch.cat(E), torch.cat(L)

    def feats_em(args):
        with Pool(a.procs) as pool:
            Ef = np.stack(pool.map(em3, args, chunksize=20))
        En, la = ft.time_features(torch.tensor(Ef, dtype=torch.complex128, device=dev), N_TIME)
        return En, la

    # GATE: empymod (no layer) == forward_torch
    g = slice(0, a.n_gate)
    Eg, Lg = feats_em([(p[i, 0], p[i, 1], p[i, 2], p[i, 3], 0.0, 0.0, p[i, 1]) for i in range(a.n_gate)])
    Ef, Lf = feats_ft(p[g])
    gE = float((Eg.float() - Ef).abs().max()); gL = float((Lg.float() - Lf).abs().max())
    gE64 = float((Eg - feats_ft_64(ft, torch, dev, p[g])[0]).abs().max())
    print(f'GATE empymod vs forward_torch: max|dEnorm| {gE64:.2e} (float32 {gE:.1e}), max|dlogamp| {gL:.1e}', flush=True)
    if gE64 > 1e-8:
        raise SystemExit('GATE FAILED')
    E3, L3 = feats_em([(p[i, 0], p[i, 1], p[i, 2], p[i, 3], b[i], h[i], s3[i]) for i in range(a.n)])
    E3, L3 = E3.float(), L3.float()
    E2, L2 = feats_ft(p)
    dla = (L3 - L2).abs().cpu().numpy(); dtr = torch.sqrt(((E3 - E2) ** 2).mean(-1)).cpu().numpy()
    print(f'3L data built n={a.n} {time.time()-t0:.0f}s; median |dlogamp| {np.median(dla):.4f} dec, p90 {np.percentile(dla,90):.4f}; '
          f'median trace rms change {np.median(dtr):.4f}', flush=True)

    def build(run):
        m = ARCHS[a.arch](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
        m.load_state_dict(torch.load(os.path.join(a.root, run, 'best.pt'), map_location=dev, weights_only=True))
        return m.eval()

    @torch.no_grad()
    def evaluate(m, E, L):
        P, tr, am = [], [], []
        for i in range(0, len(E), a.batch):
            x, _ = make_x(E[i:i + a.batch], L[i:i + a.batch], False)
            pn = m(x)[1].float()
            pp = (BOUNDS[:, 0].to(dev) + pn * (BOUNDS[:, 1] - BOUNDS[:, 0]).to(dev)).float()
            t, s = misfit_terms(pp, E[i:i + a.batch], L[i:i + a.batch])
            P.append(pn.cpu()); tr.append(t.cpu()); am.append(s.cpu())
        return torch.cat(P).double().numpy(), torch.cat(tr).double().numpy(), torch.cat(am).double().numpy()

    def r2(y, q):
        return (1 - ((y - q) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).round(4).tolist()

    def auc(pos, neg):
        s = np.concatenate([pos, neg]); _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    res = dict(runs=a.runs.split(','), n=a.n, seed=a.seed, order=NAMES, gate=dict(max_dEnorm=gE64, max_dlogamp=gL),
               design=dict(burial_m=[0, 50], thickness_m=[5, 50], contrast=[10, 100], target_sigma2='host'),
               layer_effect=dict(median_abs_dlogamp=float(np.median(dla)), p90_abs_dlogamp=float(np.percentile(dla, 90)),
                                 median_trace_rms_change=float(np.median(dtr))), per_run={})
    for run in res['runs']:
        m = build(run); out = {}
        for name, (E, L) in (('control', (E2, L2)), ('three_layer', (E3, L3))):
            P, tr, am = evaluate(m, E, L)
            err = np.sqrt(((P - th) ** 2).sum(1)); mis = tr + am
            out[name] = dict(r2=r2(th, P), rmse=np.sqrt(((P - th) ** 2).mean(0)).round(4).tolist(),
                             trace_rmse=float(np.sqrt(tr.mean())), logamp_rmse=float(np.sqrt(am.mean())),
                             misfit_median=float(np.median(mis)), spearman_misfit_vs_error=float(spearmanr(mis, err)[0]), _m=mis)
        out['auc_3L_vs_control'] = auc(out['three_layer']['_m'], out['control']['_m'])
        del out['control']['_m'], out['three_layer']['_m']
        res['per_run'][run] = out
        print(run, json.dumps(out), flush=True)
    json.dump(res, open(os.path.join(a.out, f'o2_{a.tag}.json'), 'w'), indent=1)
    print('DONE', f'{time.time()-t0:.0f}s', flush=True)


def feats_ft_64(ft, torch, dev, pp):
    q = torch.tensor(pp, dtype=torch.float64, device=dev)
    return ft.time_features(ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6), N_TIME)


if __name__ == '__main__':
    main()
