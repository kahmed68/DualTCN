"""
forward_al611.py -- differentiable PyTorch forward operator for the AL611 (SWAN) geometry (G3, 2026-11-13).

Model (1D, isotropic):
    layer 0 : air                        z < 0              res RES_AIR
    layer 1 : seawater                   0 < z < hw         sigma_w
    layers 2..K+1 : sediment layers      thicknesses h_1..h_{K-1}, last one is a half-space
Source : x-directed finite electric bipole of length L, centred at x = 0, depth zs (in the water layer),
         integrated with n_src Gauss-Legendre points and normalised to unit moment (empymod strength=0).
Receivers : inline point Ex dipoles at (offset_j, 0, zr_j), also in the water layer.
Signal : step-off (switch-off) response E(t) at the gate times, empymod signal=-1:
    E(t) = 2/pi * 1/t * sum_k Re[ -E(f_k) / (i w_k) ] * cos_k ,  w_k = base_k / t  (standard Fourier DLF)
Hankel : standard DLF, Key (2012) 201-point filter (key_201_2009.npz), as in forward_torch.py.

The frequency kernel is forward_torch.freq_response generalised to a stack of layers below the water: the
bottom reflection coefficient of the water layer is the recursive (generalised) reflection coefficient
(empymod.kernel.reflections). Everything else follows empymod.kernel.greenfct/fields for lsrc = lrec = 1.

Two Fourier modes:
    exact  : evaluates the kernel at all DLF frequencies (n_times x 201); reference for validation.
    interp : evaluates on a log-frequency grid with `ppd` points per decade and interpolates
             Re[-E/(i w)] (cubic Hermite in log f), then applies the same DLF. Much cheaper.
All computation in float64/complex128; gradients flow to every model parameter.
"""
import math
import os

import numpy as np
import torch

MU0 = 4e-7 * math.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 * C0)
RES_AIR = 2e14

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HANKEL = os.path.join(_SRC, "key_201_2009.npz")
_FOURIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key_201_2012_cos.npz")
_CACHE = {}


def _filters(device):
    k = str(device)
    if k not in _CACHE:
        h = np.load(_HANKEL)
        if not os.path.exists(_FOURIER):
            import empymod
            fo = empymod.filters.Fourier().key_201_2012
            np.savez(_FOURIER, base=fo.base, cos=fo.cos)
        f = np.load(_FOURIER)
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        _CACHE[k] = dict(hb=t(h["base"]), j0=t(h["j0"]), j1=t(h["j1"]), fb=t(f["base"]), fc=t(f["cos"]))
    return _CACHE[k]


def freq_response(sig_w, hw, res_sed, thk_sed, offsets, zs, zr, freqs, res_air=RES_AIR):
    """Inline Ex of a unit x-directed point HED at depth zs, receiver at depth zr (both in the water layer).

    sig_w (B,) [S/m]; hw (B,) water depth [m]; res_sed (B,K) [ohm-m]; thk_sed (B,K-1) [m];
    offsets (R,) tensor [m]; zs, zr floats [m]; freqs (F,) tensor [Hz]. Returns (B,R,F) complex128.
    """
    fl = _filters(sig_w.device)
    base, j0, j1 = fl["hb"], fl["j0"], fl["j1"]
    s = 2j * math.pi * freqs                                   # (F,)
    zeta = s * MU0
    lam = base[None, :] / offsets[:, None]                     # (R,N)
    l2 = (lam * lam)[None, :, None, :]                         # (1,R,1,N)

    def gam(eta):                                              # eta (B,F) or (1,F) -> (B,R,F,N)
        return torch.sqrt(l2 + (zeta[None, :] * eta)[:, None, :, None])

    eta0 = (1.0 / res_air + s * EPS0)[None, :]
    eta1 = sig_w[:, None] + s[None, :] * EPS0
    etas = [1.0 / res_sed[:, k:k + 1] + s[None, :] * EPS0 for k in range(res_sed.shape[1])]
    G0, G1 = gam(eta0), gam(eta1)
    Gs = [gam(e) for e in etas]
    E0, E1 = eta0[:, None, :, None], eta1[:, None, :, None]
    Es = [e[:, None, :, None] for e in etas]
    Z = zeta[None, None, :, None]

    dsf = hw[:, None, None, None]
    e_dsf = torch.exp(-2.0 * G1 * dsf)
    Wu = torch.exp(-G1 * (dsf - zr))
    Wd = torch.exp(-G1 * zr)
    direct = torch.exp(-G1 * abs(zs - zr))
    K = len(Gs)

    greens = {}
    for TM in (True, False):
        if TM:
            e0, e1, es, pm = E0, E1, Es, -1.0
        else:
            e0, e1, es, pm = Z, Z, [Z] * K, 1.0
        # generalised reflection coefficient at the water bottom, recursion from the deepest interface up
        Rb = None
        for k in range(K - 1, 0, -1):                          # interface between sediment k-1 and k
            ga, gb, ea, eb = Gs[k - 1], Gs[k], es[k - 1], es[k]
            r = (eb * ga - ea * gb) / (eb * ga + ea * gb)
            if Rb is not None:
                ex = torch.exp(-2.0 * gb * thk_sed[:, k, None, None, None])
                r = (r + Rb * ex) / (1.0 + r * Rb * ex)
            Rb = r
        r = (es[0] * G1 - e1 * Gs[0]) / (es[0] * G1 + e1 * Gs[0])   # water / first sediment
        if Rb is not None:
            ex = torch.exp(-2.0 * Gs[0] * thk_sed[:, 0, None, None, None])
            r = (r + Rb * ex) / (1.0 + r * Rb * ex)
        Rp = r
        Rm = (e0 * G1 - e1 * G0) / (e0 * G1 + e1 * G0)         # air / water
        den = 1.0 - Rm * Rp * e_dsf
        Pd = (torch.exp(-G1 * zs) + pm * Rp * torch.exp(-G1 * (2 * dsf - zs))) * Rm / den
        Pu = (torch.exp(-G1 * (dsf - zs)) + pm * Rm * torch.exp(-G1 * (dsf + zs))) * Rp / den
        g = Pu * Wu + Pd * Wd + (-direct if TM else direct)
        greens[TM] = g * G1 / E1 if TM else g * Z / G1

    GTM, GTE = greens[True], greens[False]
    lamb = lam[None, :, None, :]
    Ptot = (GTM + GTE) / (4.0 * math.pi)
    out = (-Ptot) @ j1.to(torch.complex128) / offsets[None, :, None]
    out = out + (0.5 * Ptot * lamb) @ j0.to(torch.complex128)
    out = out + ((GTM - GTE) * lamb / (8.0 * math.pi)) @ j0.to(torch.complex128)
    return out / offsets[None, :, None]


def _gl(n, L, device):
    x, w = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(x * L / 2, dtype=torch.float64, device=device),
            torch.as_tensor(w / 2, dtype=torch.float64, device=device))


def _hermite_interp(xq, x, y):
    """Monotone-agnostic cubic Hermite (Catmull-Rom style slopes) of y(B,R,F) at xq (Q,) on uniform grid x (F,)."""
    dx = x[1] - x[0]
    m = torch.empty_like(y)
    m[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (2 * dx)
    m[..., 0] = (y[..., 1] - y[..., 0]) / dx
    m[..., -1] = (y[..., -1] - y[..., -2]) / dx
    i = torch.clamp(((xq - x[0]) / dx).floor().long(), 0, len(x) - 2)
    u = (xq - x[i]) / dx
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return h00 * y[..., i] + h10 * dx * m[..., i] + h01 * y[..., i + 1] + h11 * dx * m[..., i + 1]


def stepoff(sig_w, hw, res_sed, thk_sed, times, rx, L=100.0, zs=3.39, n_src=11, mode="interp", ppd=20,
            res_air=RES_AIR):
    """Step-off Ex(t) [V/m per A.m] for each receiver.

    rx : list of (offset [m], zr [m]) inline receivers; offset measured from the bipole centre.
    times : (T,) gate times [s]. Returns (B, len(rx), T) float64.
    """
    dev = sig_w.device
    fl = _filters(dev)
    fb, fc = fl["fb"], fl["fc"]
    t = torch.as_tensor(times, dtype=torch.float64, device=dev)
    xs, ws = _gl(n_src, L, dev)
    w_all = (fb[None, :] / t[:, None])                          # (T,201) angular frequencies
    out = []
    for off, zr in rx:
        roff = off - xs                                         # (n_src,) point-source offsets
        if mode == "exact":
            f = (w_all / (2 * math.pi)).reshape(-1)
            Ef = freq_response(sig_w, hw, res_sed, thk_sed, roff, zs, zr, f, res_air)   # (B,n_src,T*201)
            Ef = (Ef * ws[None, :, None]).sum(1)                                        # (B,T*201)
            g = (-Ef / (1j * 2 * math.pi * f)).real.reshape(-1, *w_all.shape)          # (B,T,201)
        else:
            lw = torch.log10(w_all / (2 * math.pi))
            lo, hi = math.floor(lw.min().item() * ppd) / ppd - 2.0 / ppd, math.ceil(lw.max().item() * ppd) / ppd + 2.0 / ppd
            lg = torch.arange(lo, hi + 0.5 / ppd, 1.0 / ppd, dtype=torch.float64, device=dev)
            f = 10 ** lg
            Ef = freq_response(sig_w, hw, res_sed, thk_sed, roff, zs, zr, f, res_air)
            Ef = (Ef * ws[None, :, None]).sum(1)                                        # (B,F)
            y = (-Ef / (1j * 2 * math.pi * f)).real
            # interpolate f*y (smoother: ~ step response amplitude) and divide back
            gq = _hermite_interp(lw.reshape(-1), lg, (y * f)[:, None, :])[:, 0, :] / 10 ** lw.reshape(-1)
            g = gq.reshape(-1, *w_all.shape)
        out.append((g * fc).sum(-1) / t * (2.0 / math.pi))
    return torch.stack(out, 1)
