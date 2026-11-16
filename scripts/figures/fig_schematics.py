"""Schematic figures of the revised manuscript (R1.8, R1.9, R1.12), drawn with the common style file.
fig1a_geometry.png  survey geometry (Figure 1a)
fig1b_profile.png   layered conductivity profile produced by the decoder (Figure 1b)
fig2_architecture.png  DualTCN architecture, layer sizes from src/model_p9d.py (379,370 parameters)
Usage: python fig_schematics.py <outdir>"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, 'dualtcn.mplstyle'))
OUT = sys.argv[1]
BLUE, VERM, GREEN, PINK, ORANGE, SKY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9'


# ---------------------------------------------------------------- Figure 1a: geometry (schematic, not to scale)
def fig1a():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.set_xlim(-1.3, 7.2); ax.set_ylim(-5.9, 1.25); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 7, 1.0, color='0.93', lw=0))
    ax.add_patch(Rectangle((0, -4.2), 7, 4.2, color='#DDEFF8', lw=0))
    ax.add_patch(Rectangle((0, -5.4), 7, 1.2, color='#EADBC8', lw=0))
    ax.plot([0, 7], [0, 0], color=BLUE, lw=1.2); ax.plot([0, 7], [-4.2, -4.2], color='#8C6D46', lw=1.2)
    ax.text(0.15, 0.5, r'Air, $\rho \to \infty$', va='center', fontsize=8, color='0.3')
    ax.text(0.15, -2.0, 'Seawater\n' + r'$\sigma_1$', va='center', fontsize=8, color='#1F4E79')
    ax.text(0.15, -4.8, 'Seafloor half-space\n' + r'$\sigma_2$', va='center', fontsize=8, color='#5C4630')
    ax.text(7.05, 0.0, r'$z=0$', va='center', fontsize=7.5, color=BLUE)
    src = (1.9, -3.0)
    ax.plot(*src, 'o', color=VERM, ms=6, zorder=5)
    ax.annotate('', xy=(2.45, -3.0), xytext=(1.35, -3.0), arrowprops=dict(arrowstyle='<->', color=VERM, lw=1.2))
    ax.text(1.75, -3.45, 'HED source', ha='right', va='top', fontsize=7.5, color=VERM)
    rx = [(2.6, '20 m'), (3.4, '50 m'), (4.6, '100 m'), (6.4, '200 m')]
    zr = -0.55
    for x, lab in rx:
        ax.add_patch(Polygon([(x - 0.13, zr + 0.17), (x + 0.13, zr + 0.17), (x, zr - 0.13)], color=GREEN, lw=0))
        ax.text(x, zr + 0.24, lab, ha='center', fontsize=6.5, color=GREEN)
    ax.text(4.3, 0.55, r'Receivers at $z_\mathrm{obs} = 20$ m (offsets)', ha='center', fontsize=7.5, color=GREEN)
    for x, _ in rx[:2]:
        ax.plot([src[0], x], [src[1], zr - 0.1], color=ORANGE, ls='--', lw=0.9)
    for x, _ in rx[2:]:
        ax.plot([src[0], src[0], x, x], [src[1], -4.2, -4.2, zr - 0.1], color=PINK, ls=(0, (1, 1.6)), lw=1.5, zorder=4)
    # depth arrows
    ax.annotate('', xy=(-0.35, -3.0), xytext=(-0.35, 0), arrowprops=dict(arrowstyle='<->', color='0.2', lw=0.9))
    ax.text(-0.5, -1.5, r'$d_1$', ha='right', va='center', fontsize=9)
    ax.annotate('', xy=(-0.35, -4.2), xytext=(-0.35, -3.0), arrowprops=dict(arrowstyle='<->', color='0.2', lw=0.9))
    ax.text(-0.5, -3.6, r'$d_2$', ha='right', va='center', fontsize=9)
    # legend below the section, outside the layers
    ax.plot([0.3, 0.9], [-5.75, -5.75], color=ORANGE, ls='--', lw=0.9)
    ax.text(1.0, -5.75, 'Direct path', va='center', fontsize=7)
    ax.plot([3.4, 4.0], [-5.75, -5.75], color=PINK, ls=(0, (1, 1.6)), lw=1.5)
    ax.text(4.1, -5.75, 'Path along the seafloor', va='center', fontsize=7)
    fig.savefig(os.path.join(OUT, 'fig1a_geometry.png')); plt.close(fig)


# ---------------------------------------------------------------- Figure 1b: layered profile
def fig1b():
    fig, ax = plt.subplots(figsize=(1.9, 3.0))
    dsf = 120
    ax.plot([2.0, 2.0, 0.05, 0.05], [0, dsf, dsf, 250], color=BLUE, lw=1.6)
    ax.axhline(0, color=BLUE, lw=0.8, ls=':'); ax.axhline(dsf, color='#8C6D46', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_xlim(0.01, 10); ax.set_ylim(250, -10)
    ax.text(2.4, 60, r'$\sigma_1$', fontsize=9, color=BLUE)
    ax.text(0.065, 190, r'$\sigma_2$', fontsize=9, color=BLUE)
    ax.text(0.012, dsf - 5, r'$d_\mathrm{sf} = d_1 + d_2$', fontsize=7, color='#5C4630')
    ax.set_xlabel(r'$\sigma(z)$ (S/m)'); ax.set_ylabel('Depth $z$ (m)')
    ax.grid(False)
    fig.savefig(os.path.join(OUT, 'fig1b_profile.png')); plt.close(fig)


# ---------------------------------------------------------------- Figure 2: architecture
def box(ax, x, y, w, h, text, fc, fs=6.3, ec='0.35'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, linespacing=1.15)


def arrow(ax, p, q, ls='-', color='0.3'):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8, ls=ls,
                                                    shrinkA=0, shrinkB=0, mutation_scale=7))


def panel(ax, x0, y0, x1, y1, title, fc, ec, tc, fs=9, dashed=False, title_y=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.02,rounding_size=0.35',
                                fc=fc, ec=ec, lw=1.0, ls='--' if dashed else '-'))
    if title:
        ax.text(x0 + 0.35 if dashed else (x0 + x1) / 2, title_y if title_y else y1 - 0.55, title,
                ha='left' if dashed else 'center', va='center', fontsize=fs, color=tc, weight='bold')


def fig2():
    """DualTCN architecture in the layout of the original figure, with the corrected architecture."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.set_xlim(0, 31); ax.set_ylim(-4.75, 20); ax.axis('off')
    FT = ['#E3EEF9', '#D2E4F5', '#BFD8F1', '#A9CAEC', '#8EB8E5', '#6FA3DC']
    LT = ['#E9E2F3', '#DCD1EE', '#CDBFE7', '#BCAADF']
    EDGE = '0.45'
    # ---------------- input
    panel(ax, 0.3, 6.2, 3.4, 15.6, 'Input', '0.96', '0.6', '0.25')
    # real normalized traces of the generator (sample dataset of the release): a peak in the first
    # sample and the ringing of the 2 Hz band limit (Supplementary Section S12.8)
    t = np.arange(128) / 127.0
    smp = os.environ.get('DUALTCN_SAMPLE', 'data/sample_v6_5000.npz' if os.path.exists('data/sample_v6_5000.npz')
                         else '/work/hdd/bgte/kahmed2/DualTCN_release_r1/upload/sample_v6_5000.npz')
    E = np.load(smp)['E_multi']
    traces = [E[0, 0], E[0, 1], E[0, 2], E[1, 0], E[1, 3]]
    for k, y in enumerate([13.6, 12.4, 11.2, 8.9, 7.7]):
        w = traces[k] / traces[k][np.argmax(np.abs(traces[k]))]
        ax.plot(0.75 + 2.2 * t, y - 0.3 + 0.75 * w, color='#3B7DC4', lw=0.8)
    ax.text(1.85, 10.1, '$\\vdots$', ha='center', va='center', fontsize=11, color='#3B7DC4')
    ax.text(1.85, 5.2, '$(B, 8, 128)$\n4 receivers $\\times$\n2 channels', ha='center', va='center', fontsize=7.5, color='0.25')
    # ---------------- encoder panel
    panel(ax, 3.9, 1.9, 19.2, 19.6, 'DualTCN Encoder', '#F2F7FC', '#9DB9D8', '#1F4E79', fs=10)
    # full-time branch
    panel(ax, 4.2, 12.6, 18.9, 18.3, 'Full-Window Branch (all 128 samples)', '#F7FAFE', '#AFC6E0', '#1F4E79', fs=7.5,
          dashed=True, title_y=17.75)
    box(ax, 5.35, 14.9, 1.75, 2.9, '$1{\\times}1$ conv\n$8 \\to 64$', '0.94', fs=6.0)
    for i, d in enumerate([1, 2, 4, 8, 16, 32]):
        box(ax, 7.15 + 1.38 * i, 14.9, 1.2, 2.9, f'B{i+1}\n\nTCN\nblock\n\n$d$ = {d}', FT[i], fs=6.5)
    box(ax, 17.55, 14.9, 2.2, 2.9, 'Pool + FC\n\n$\\mathbf{z}_\\mathrm{full}$\n$(B, 256)$', '0.94', fs=6.5)
    arrow(ax, (3.4, 14.9), (4.48, 14.9)); arrow(ax, (6.23, 14.9), (6.55, 14.9))
    for i in range(5):
        arrow(ax, (7.75 + 1.38 * i, 14.9), (7.93 + 1.38 * i, 14.9))
    arrow(ax, (14.95, 14.9), (16.45, 14.9))
    # late-time branch
    panel(ax, 4.2, 7.0, 18.9, 12.2, 'Late-Window Branch (last 64 samples)', '#FAF8FD', '#C3B5DE', '#4B3A7A', fs=7.5,
          dashed=True, title_y=11.65)
    box(ax, 5.35, 9.2, 1.75, 2.9, 'Slice\n\n$t$ = 65\nto 128', '0.94', fs=6.5)
    box(ax, 7.4, 9.2, 1.75, 2.9, '$1{\\times}1$ conv\n$8 \\to 32$', '0.94', fs=6.0)
    for i, d in enumerate([1, 2, 4, 8]):
        box(ax, 9.6 + 1.5 * i, 9.2, 1.3, 2.9, f'L{i+1}\n\nTCN\nblock\n\n$d$ = {d}', LT[i], fs=6.5)
    box(ax, 17.55, 9.2, 2.2, 2.9, 'Pool + FC\n\n$\\mathbf{z}_\\mathrm{late}$\n$(B, 128)$', '#DCD1EE', fs=6.5)
    arrow(ax, (3.4, 9.2), (4.48, 9.2)); arrow(ax, (6.23, 9.2), (6.53, 9.2)); arrow(ax, (8.28, 9.2), (8.95, 9.2))
    for i in range(3):
        arrow(ax, (10.25 + 1.5 * i, 9.2), (10.45 + 1.5 * i, 9.2))
    arrow(ax, (14.75, 9.2), (16.45, 9.2))
    # residual block
    panel(ax, 4.2, 2.2, 18.9, 6.6, 'Each TCN block (dilation $d$)', '#FBFBFB', '#BDBDBD', '0.3', fs=7.5,
          dashed=True, title_y=6.05)
    yb = 3.9
    items = [('Dilated causal\nConv1d ($k$ = 3)', 3.0, '#D2E4F5'), ('BatchNorm', 2.3, '#D5EFE3'),
             ('GELU', 1.3, '#FBE3CC'), ('Dropout\n($p$ = 0.3)', 1.9, '#FDEBD0')]
    x = 5.3; xs = []
    for k, (txt, w, c) in enumerate(items):
        if k == 3:
            x += 1.0
        box(ax, x + w / 2, yb, w, 1.35, txt, c, fs=6.5); xs.append((x, x + w)); x += w + 0.45
    for a_, b_ in zip(xs[:2], xs[1:3]):
        arrow(ax, (a_[1], yb), (b_[0], yb))
    arrow(ax, (xs[2][1], yb), (xs[3][0], yb))
    # bracket "x 2" over the first three boxes
    bx0, bx1 = xs[0][0], xs[2][1]
    ax.plot([bx0, bx0, bx1, bx1], [yb + 0.85, yb + 1.1, yb + 1.1, yb + 0.85], color='0.35', lw=0.8)
    ax.text((bx0 + bx1) / 2, yb + 1.35, 'repeated twice', ha='center', fontsize=6.5, color='0.3')
    xp = xs[3][1] + 0.75
    ax.add_patch(plt.Circle((xp, yb), 0.3, fc='white', ec='0.35', lw=0.9))
    ax.text(xp, yb, '+', ha='center', va='center', fontsize=10)
    arrow(ax, (xs[3][1], yb), (xp - 0.3, yb)); arrow(ax, (4.6, yb), (xs[0][0], yb))
    ax.plot([4.85, 4.85, xp], [yb, 2.65, 2.65], color='0.35', lw=0.8); arrow(ax, (xp, 2.65), (xp, yb - 0.3))
    ax.text((4.85 + xp) / 2, 2.85, 'residual connection', ha='center', fontsize=6.3, color='0.35')
    arrow(ax, (xp + 0.3, yb), (xp + 1.0, yb))
    # ---------------- heads panel
    panel(ax, 19.7, 1.9, 25.4, 19.6, 'Prediction Heads', '#F6F3FB', '#B9A9D6', '#4B3A7A', fs=10)
    box(ax, 22.55, 15.6, 4.6, 2.6, 'Stage 1: MLP\n$256 \\to 128 \\to 2$\n\n$[\\hat\\sigma_1, \\hat d_1]$', '#ECE6F5', fs=7)
    box(ax, 21.3, 11.7, 2.3, 1.6, 'Concat\n$\\mathbf{z}_\\mathrm{comb}$, $(B, 384)$', '#8EB8E5', fs=6.0)
    box(ax, 23.0, 8.0, 3.8, 2.6, 'Stage 2: MLP\n$386 \\to 193 \\to 2$\n\n$[\\hat\\sigma_2, \\hat d_2]$', '#E3EEF9', fs=7)
    box(ax, 23.0, 3.9, 3.8, 2.5, 'Auxiliary head: MLP\n$384 \\to 192 \\to 1$\n$\\hat d_\\mathrm{sf}$ (training only)', '#FFF3C4', fs=6.4)
    # full -> stage 1 and concat; late -> concat
    ax.plot([18.65, 19.45, 19.45], [14.9, 14.9, 15.6], color='0.3', lw=0.8); arrow(ax, (19.45, 15.6), (20.25, 15.6))
    ax.plot([19.45, 19.45], [14.9, 12.0], color='0.3', lw=0.8); arrow(ax, (19.45, 12.0), (20.15, 12.0))
    ax.plot([18.65, 19.75, 19.75], [9.2, 9.2, 11.4], color='0.3', lw=0.8); arrow(ax, (19.75, 11.4), (20.15, 11.4))
    ax.plot([22.45, 23.3], [11.7, 11.7], color='0.3', lw=0.8); ax.plot([23.3, 23.3], [11.7, 9.3], color='0.3', lw=0.8)
    arrow(ax, (23.3, 9.5), (23.3, 9.3))
    arrow(ax, (24.2, 14.3), (24.2, 9.3), ls='--')
    ax.text(24.35, 11.8, 'detached', rotation=90, fontsize=6.3, color='0.35', va='center')
    ax.plot([20.55, 20.55, 20.9], [10.9, 3.9, 3.9], color='0.3', lw=0.8); arrow(ax, (20.9, 3.9), (21.1, 3.9))
    # ---------------- decoder panel
    panel(ax, 25.9, 1.9, 30.8, 19.6, 'Profile Decoder', '#FFF8EF', '#E8C79A', '#9A5B13', fs=10)
    box(ax, 28.35, 15.6, 4.3, 2.6, 'Assemble\n\n$\\hat{\\mathbf{p}} \\in [0, 1]^4$', '#D5EFE3', fs=7)
    box(ax, 28.35, 9.6, 4.3, 4.6, 'Layered-profile\ndecoder, Eq. (3)\n\n$\\sigma(z) = \\tilde\\sigma_1 + (\\tilde\\sigma_2 - \\tilde\\sigma_1)$\n'
        '$\\times\\, \\sigma_\\mathrm{s}\\left((z - \\tilde d_\\mathrm{sf})/\\tau\\right)$', '#FDEBD0', fs=6.6)
    box(ax, 28.35, 3.9, 4.3, 2.5, 'Profile\n\n$\\log_{10}\\hat\\sigma(z)$\n64 depth nodes', '#DCEFF6', fs=7)
    arrow(ax, (24.85, 15.6), (26.2, 15.6))
    ax.plot([24.9, 25.65, 25.65], [8.0, 8.0, 15.0], color='0.3', lw=0.8); arrow(ax, (25.65, 15.0), (26.2, 15.0))
    arrow(ax, (28.35, 14.3), (28.35, 11.9)); arrow(ax, (28.35, 7.3), (28.35, 5.15))
    # ---------------- forward-operator strip (11-16, title "... with a Differentiable Forward Operator")
    panel(ax, 0.3, -3.2, 30.8, 1.55, 'Differentiable Forward Operator (Section 2.2)', '#F3FAF6', '#93C9AE', '#1D6B47',
          fs=9, title_y=1.0)
    yo = -1.0
    box(ax, 1.85, yo, 2.6, 2.2, 'Observed\ninput $\mathbf{x}$', '0.94', fs=7)
    box(ax, 6.0, yo, 3.4, 2.2, 'Predicted\nparameters $\hat{\mathbf{p}}$', '#D5EFE3', fs=7)
    box(ax, 11.4, yo, 5.4, 2.2, 'Layered-earth solution\n(PyTorch, Hankel digital\nfilter, float64)', '#E2F3EA', fs=7)
    box(ax, 16.85, yo, 3.3, 2.2, 'Simulated\ninput $\hat{\mathbf{x}}$', '0.94', fs=7)
    box(ax, 21.3, yo, 3.4, 2.2, 'Data misfit\n(trace and\nlog-amplitude)', '#FFF3C4', fs=7)
    box(ax, 27.55, -0.2, 5.6, 1.3, 'Check of every prediction\n(Sections 3.5 and 3.6)', '#DCEFF6', fs=6.4)
    box(ax, 27.55, -1.85, 5.6, 1.3, 'Loss $\mathcal{L}_\mathrm{fc}$ of the forward-\nconsistent variant (Section 2.5)',
        '#FDEBD0', fs=6.4, ec='0.5')
    arrow(ax, (1.85, 4.35), (1.85, 0.1))
    arrow(ax, (7.7, yo), (8.7, yo)); arrow(ax, (14.1, yo), (15.2, yo)); arrow(ax, (18.5, yo), (19.6, yo))
    ax.plot([1.85, 1.85, 21.3], [-2.1, -2.75, -2.75], color='0.3', lw=0.8); arrow(ax, (21.3, -2.75), (21.3, -2.1))
    ax.plot([23.0, 24.0, 24.0], [yo, yo, -0.2], color='0.3', lw=0.8); arrow(ax, (24.0, -0.2), (24.75, -0.2))
    ax.plot([24.0, 24.0], [yo, -1.85], color='0.3', lw=0.8, ls='--'); arrow(ax, (24.0, -1.85), (24.75, -1.85), ls='--')
    # ---------------- bottom bar
    ax.add_patch(FancyBboxPatch((3.0, -4.55), 25.0, 1.15, boxstyle='round,pad=0.02,rounding_size=0.3',
                                fc='#F4F8FC', ec='#C9D9EA', lw=0.8))
    ax.text(15.5, -3.98, 'DualTCN   |   379,370 parameters   |   1.7 ms per sample on one CPU core (batched inference)',
            ha='center', va='center', fontsize=8, color='0.2')
    fig.savefig(os.path.join(OUT, 'fig2_architecture.png')); plt.close(fig)


fig1a(); fig1b(); fig2()
print('done')
