#!/usr/bin/env python3
"""C1: conventional inversion against DualTCN on the corrected (v6) data (EMS revision, R2.7 and R4.4; 2026-11-13).

Scope (Khaled 11-13): CLEAN test data only, one fixed damping value, v1 protocol otherwise.
Methods, all through the same exact differentiable forward operator (forward_torch, complex128) with EXACT
Jacobians (torch autograd, forward mode for LM), parameters theta = normalised log10 (sigma1, sigma2, d1, d2) in [0,1]^4 as the network:
  LM          scipy least_squares method='lm' (unbounded; non-finite or invalid d1 <= Z_OBS -> large residual), 8 starts
  LBFGSB      scipy minimize L-BFGS-B, bounds [0,1]^4, 8 starts
  WS-LM, WS-LBFGSB  warm start: start 0 = DualTCN prediction (base_s1), starts 1-7 = the same random starts
Starts (v1 protocol): midpoint + 7 random U[0,1]^4, drawn once and shared by all samples.
Objective (v1): sum over receivers of ||(E_obs - E_pred)/||E_obs|| ||^2 on the raw transients E = E_norm * 10^log_amp,
  plus damping toward the prior mean: lam * ||theta - 0.5||^2 (W = identity in normalised space = inverse prior width).
  lam = 1e-6, fixed before the run (negligible bias on clean data; only stabilises flat directions).
Tolerances (v1): LM max_nfev 2000, ftol = xtol = 1e-6; L-BFGS-B maxiter 300, ftol 1e-10, gtol 1e-7.
Best start = lowest objective. Final theta clipped to [0,1] for the metrics (the network output is in [0,1]).
Outputs: per-parameter R2 (normalised space, same as the network tables), trace RMSE and log-amp RMSE of the
re-simulated data (same definitions as train_v6), objective, nfev/njev, wall time per sample (1 CPU thread),
convergence history (objective per function evaluation, best start) for the first --n-hist samples.
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--starts', type=int, default=8)
ap.add_argument('--lam', type=float, default=1e-6)
ap.add_argument('--net-run', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/e1/base_s1')
ap.add_argument('--workers', type=int, default=32)
ap.add_argument('--n-hist', type=int, default=20)
ap.add_argument('--snr', type=float, default=0.0, help='11-14 (R2.7): white noise on the raw transient at this peak SNR in dB (0 = clean), as in N1')
ap.add_argument('--noise-seed', type=int, default=4040)
ap.add_argument('--methods', default='LM,LBFGSB,WS-LM,WS-LBFGSB', help='11-14: subset of methods to run')
ap.add_argument('--split', default='test', choices=['test', 'val'], help='11-14: val = choose lambda on validation samples')
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/c1')
a = ap.parse_args()
METHODS = set(a.methods.split(','))
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')
sys.path.insert(0, SRC)
import torch
import scipy.optimize
from config_v6 import (RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, TRAIN_SPLIT, VAL_SPLIT, DATA_PATH_V6,
                       LOG_SIGMA1_MIN, LOG_SIGMA1_MAX, LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
                       LOG_D1_MIN, LOG_D1_MAX, LOG_D2_MIN, LOG_D2_MAX)
import forward_torch as ft

LO = np.array([LOG_SIGMA1_MIN, LOG_SIGMA2_MIN, LOG_D1_MIN, LOG_D2_MIN], float)
HI = np.array([LOG_SIGMA1_MAX, LOG_SIGMA2_MAX, LOG_D1_MAX, LOG_D2_MAX], float)
LO_t, RG_t = torch.tensor(LO), torch.tensor(HI - LO)
MID = np.full(4, 0.5)


def feats_t(th):                                  # (4,) float64 tensor -> E_norm (R,T), log_amp (R,)
    p = 10 ** (LO_t + th * RG_t)
    En, la = ft.time_features(ft.freq_response(p[0:1], p[1:2], p[2:3], p[3:4], RECEIVER_OFFSETS, Z_OBS, FREQS_V6), N_TIME)
    return En[0], la[0]


class Problem:
    def __init__(self, En_obs, la_obs, lam):
        raw = En_obs.astype(np.float64) * 10.0 ** la_obs.astype(np.float64)[:, None]
        self.raw = torch.tensor(raw); self.scale = self.raw.norm(dim=1, keepdim=True) + 1e-60
        self.En_obs = torch.tensor(En_obs, dtype=torch.float64); self.la_obs = torch.tensor(la_obs, dtype=torch.float64)
        self.sl = float(np.sqrt(lam)); self.hist = None; self.n_res = self.raw.numel() + 4

    def res_t(self, th):
        En, la = feats_t(th)
        r = ((self.raw - En * 10 ** la[:, None]) / self.scale).reshape(-1)
        return torch.cat([r, self.sl * (th - 0.5)])

    def res(self, x):
        try:                                      # 11-14: unbounded LM can step to d1 <= Z_OBS (receivers below the
            with torch.no_grad():                 # source depth), which forward_torch rejects; treat as invalid model
                r = self.res_t(torch.tensor(x, dtype=torch.float64)).numpy()
        except ValueError:
            r = np.full(self.n_res, np.nan)
        if not np.all(np.isfinite(r)):
            r = np.full(r.shape, 1e3)
        if self.hist is not None:
            self.hist.append(float(r @ r))
        return r

    def jac(self, x):
        # 11-14: forward mode (4 parameters -> 4 JVPs). Same exact Jacobian as the earlier reverse-mode vectorised
        # call (max rel diff 2.5e-16..6.7e-16 on 3 test samples) but ~0.15 s instead of ~60 s per call.
        try:
            J = torch.func.jacfwd(self.res_t)(torch.tensor(x, dtype=torch.float64)).numpy()
        except ValueError:
            return np.zeros((self.n_res, 4))
        return J if np.all(np.isfinite(J)) else np.zeros_like(J)

    def fg(self, x):                              # objective and gradient for L-BFGS-B
        th = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        try:
            r = self.res_t(th)
        except ValueError:
            return 1e6, np.zeros(4)
        f = (r * r).sum(); f.backward()
        fv, g = float(f), th.grad.numpy().copy()
        if not np.isfinite(fv) or not np.all(np.isfinite(g)):
            return 1e6, np.zeros(4)
        if self.hist is not None:
            self.hist.append(fv)
        return fv, g

    def misfit(self, x):                          # train_v6 definitions: MSE of E_norm and of log_amp
        with torch.no_grad():
            En, la = feats_t(torch.tensor(np.clip(x, 0, 1), dtype=torch.float64))
        return float(((En - self.En_obs) ** 2).mean()), float(((la - self.la_obs) ** 2).mean())


def run_lm(P, x0):
    r = scipy.optimize.least_squares(P.res, x0, jac=P.jac, method='lm', max_nfev=2000, ftol=1e-6, xtol=1e-6)
    return r.x, float(r.cost * 2), int(r.nfev), int(r.njev or 0)


def run_lb(P, x0):
    r = scipy.optimize.minimize(P.fg, x0, jac=True, method='L-BFGS-B', bounds=[(0, 1)] * 4,
                                options=dict(maxiter=300, ftol=1e-10, gtol=1e-7))
    return r.x, float(r.fun), int(r.nfev), int(r.nfev)


def invert(args):
    i, En_obs, la_obs, starts, net0, keep_hist = args
    torch.set_num_threads(1)
    P = Problem(En_obs, la_obs, a.lam)
    out = {}
    for name, fn, s0 in (('LM', run_lm, starts), ('LBFGSB', run_lb, starts),
                         ('WS-LM', run_lm, np.vstack([net0, starts[1:]])), ('WS-LBFGSB', run_lb, np.vstack([net0, starts[1:]]))):
        if name not in METHODS:
            continue
        t0 = time.perf_counter(); best = None; nfev = njev = 0; per = []
        for k, x0 in enumerate(s0):
            P.hist = [] if keep_hist else None
            x, f, nf, nj = fn(P, x0.copy())
            nfev += nf; njev += nj; per.append(f)
            if best is None or f < best[1]:
                best = (x, f, P.hist)
        P.hist = None
        tr, am = P.misfit(best[0])
        out[name] = dict(theta=np.clip(best[0], 0, 1).tolist(), theta_raw=best[0].tolist(), obj=best[1], obj_starts=per,
                         trace_mse=tr, logamp_mse=am, nfev=nfev, njev=njev, time=time.perf_counter() - t0,
                         hist=best[2] if keep_hist else None)
    tr, am = P.misfit(net0)
    out['DualTCN'] = dict(theta=np.clip(net0, 0, 1).tolist(), trace_mse=tr, logamp_mse=am,
                          obj=float(P.res(np.asarray(net0)) @ P.res(np.asarray(net0))))
    return i, out


def r2(y, p):
    return (1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).round(4).tolist()


if __name__ == '__main__':
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d = np.load(DATA_PATH_V6)
    n = len(d['params']); n_tr = int(n * TRAIN_SPLIT); n_va = int(n * VAL_SPLIT)
    rng = np.random.default_rng(a.seed)
    pool_idx = np.arange(n_tr + n_va, n) if a.split == 'test' else np.arange(n_tr, n_tr + n_va)
    idx = np.sort(rng.choice(pool_idx, a.n, replace=False))
    En = d['E_multi'][idx]; La = d['log_amps'][idx]; Pp = d['params'][idx].astype(np.float64)
    y = np.clip((np.log10(Pp + 1e-12) - LO) / (HI - LO + 1e-8), 0, 1)
    if a.snr > 0:                                 # N1 definition: raw = E_norm*10^L; + peak*10^(-snr/20)*N(0,1); renormalise
        g = np.random.default_rng(a.noise_seed)
        raw = En.astype(np.float64) * 10.0 ** La.astype(np.float64)[..., None]
        pk = np.abs(raw).max(-1, keepdims=True)
        raw = raw + pk * 10 ** (-a.snr / 20) * g.standard_normal(raw.shape)
        pk2 = np.abs(raw).max(-1)
        En = (raw / pk2[..., None]).astype(np.float32); La = np.log10(pk2).astype(np.float32)
        print(f'noise: SNR {a.snr} dB on the raw transients (noise seed {a.noise_seed})', flush=True)
    starts = np.vstack([MID, rng.uniform(0, 1, (a.starts - 1, 4))])
    print(f'data n={n} test samples {a.n} (seed {a.seed}) {time.time()-t0:.0f}s', flush=True)

    # DualTCN predictions (base_s1 best.pt, BatchNorm statistics as saved = recalibrated), CPU timing
    from train_v6 import ARCHS
    from config_v6 import IN_CHANNELS, N_DEPTH, LATENT_DIM, DROPOUT
    torch.set_num_threads(1)
    m = ARCHS['dualtcn'](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT)
    m.load_state_dict(torch.load(os.path.join(a.net_run, 'best.pt'), map_location='cpu', weights_only=True)); m.eval()
    x = torch.stack([torch.from_numpy(En), torch.from_numpy(La)[..., None].expand(-1, -1, En.shape[-1])], 2)
    x = x.reshape(a.n, -1, En.shape[-1]).float()
    with torch.no_grad():
        m(x[:8]); tt = time.perf_counter(); net = m(x)[1].double().numpy(); t_batch = time.perf_counter() - tt
        tt = time.perf_counter(); [m(x[j:j + 1]) for j in range(min(50, a.n))]; t_single = (time.perf_counter() - tt) / min(50, a.n)
    print(f'DualTCN R2 {r2(y, np.clip(net, 0, 1))}  cpu1 batch {t_batch / a.n * 1e3:.2f} ms/sample, single {t_single * 1e3:.1f} ms', flush=True)

    from multiprocessing import get_context
    # per-sample results are appended to a JSONL file as they finish, so a run stopped by the wall time can resume
    part = os.path.join(a.out, 'c1_partial.jsonl'); res = [None] * a.n
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            if r['seed'] == a.seed and r['k'] < a.n and r['idx'] == int(idx[r['k']]):
                res[r['k']] = r['out']
    todo = [k for k in range(a.n) if res[k] is None]
    print(f'resume: {a.n - len(todo)} done, {len(todo)} to do', flush=True)
    jobs = [(k, En[k], La[k], starts, net[k], k < a.n_hist) for k in todo]
    done = a.n - len(todo)
    with get_context('fork').Pool(a.workers) as pool, open(part, 'a') as fp:
        for k, o in pool.imap_unordered(invert, jobs):
            res[k] = o; done += 1
            fp.write(json.dumps(dict(seed=a.seed, k=k, idx=int(idx[k]), out=o)) + '\n'); fp.flush()
            print(f'{done}/{a.n} {time.time()-t0:.0f}s sample {k}: ' +
                  ' '.join(f"{m} {o[m]['time']:.0f}s" for m in ('LM', 'LBFGSB', 'WS-LM', 'WS-LBFGSB') if m in o), flush=True)

    meths = ['DualTCN'] + [m for m in ('LM', 'LBFGSB', 'WS-LM', 'WS-LBFGSB') if m in METHODS]
    summ = dict(n=a.n, seed=a.seed, split=a.split, methods_run=sorted(METHODS), snr=a.snr, noise_seed=a.noise_seed, idx_first=idx[:5].tolist(), lam=a.lam, starts=starts.tolist(), net_run=a.net_run,
                order=['sigma1', 'sigma2', 'd1', 'd2'], space='normalised log10 params; theta clipped to [0,1]',
                dualtcn_cpu_ms_per_sample=dict(batch=t_batch / a.n * 1e3, single=t_single * 1e3), methods={})
    for mt in meths:
        th = np.array([r[mt]['theta'] for r in res])
        tr = np.array([r[mt]['trace_mse'] for r in res]); am = np.array([r[mt]['logamp_mse'] for r in res])
        e = dict(r2=r2(y, th), r2_mean=float(np.mean(r2(y, th))),
                 trace_rmse=float(np.sqrt(tr.mean())), logamp_rmse=float(np.sqrt(am.mean())),
                 trace_rmse_median=float(np.median(np.sqrt(tr))), logamp_rmse_median=float(np.median(np.sqrt(am))),
                 obj_median=float(np.median([r[mt]['obj'] for r in res])))
        if mt != 'DualTCN':
            tm = np.array([r[mt]['time'] for r in res])
            e.update(time_mean_s=float(tm.mean()), time_median_s=float(np.median(tm)),
                     nfev_mean=float(np.mean([r[mt]['nfev'] for r in res])),
                     frac_theta_outside_01=float(np.mean([np.any((np.array(r[mt]['theta_raw']) < 0) | (np.array(r[mt]['theta_raw']) > 1)) for r in res])))
        summ['methods'][mt] = e
        print(mt, json.dumps(e), flush=True)
    json.dump(summ, open(os.path.join(a.out, 'c1_summary.json'), 'w'), indent=1)
    json.dump(dict(idx=idx.tolist(), y=y.tolist(), results=res), open(os.path.join(a.out, 'c1_per_sample.json'), 'w'))
    print('DONE', f'{time.time()-t0:.0f}s', flush=True)
