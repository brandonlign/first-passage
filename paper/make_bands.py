"""Figure 4: the spectral polynomial h(s) of a homogeneous and an
inhomogeneous three-state ring with the same affinity (unit mean period)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def spectral_poly(p, q):
    n = len(p)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, (i + 1) % n] += p[i]; Q[i, (i - 1) % n] += q[i]; Q[i, i] -= p[i] + q[i]
    P, M = np.prod(p), np.prod(q)
    F = np.log(P / M)
    c = np.poly(Q)                       # det(sI - Q), highest power first
    c = c / (2 * np.sqrt(P * M)); c[-1] += np.cosh(F / 2)
    # rescale time so that h'(0) = sinh(F/2), i.e. E tau = 1
    lam = c[-2] / np.sinh(F / 2)
    c = c * lam ** np.arange(n, -1, -1)
    return F, np.poly1d(c)

F0 = 3.0
a = np.exp(F0 / 3)
cases = [("Homogeneous cycle (all gaps closed)", [a] * 3, [1.0] * 3),
         ("Inhomogeneous cycle (two open gaps)", [a * 1.6, a, a / 1.6], [1.0, 1.0, 1.0])]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, (title, p, q) in zip(axes, cases):
    F, h = spectral_poly(p, q)
    roots = np.sort(np.concatenate([(h - 1).r, (h + 1).r]).real)
    lo = roots[0] * 1.08
    s = np.linspace(lo, 0.12 * abs(lo), 2000)
    ax.axhspan(-1, 1, color="0.92", zorder=0)
    ax.axhline(1, color="0.5", lw=0.7); ax.axhline(-1, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.plot(s, h(s), color="C0", lw=1.8)
    band = np.abs(h(s)) <= 1
    ax.fill_between(s, -3.2, -2.9, where=band, color="C3", lw=0)
    for x in np.sort(h.deriv().r.real):
        ax.plot([x], [h(x)], "o", color="k", ms=3.5)
    ax.set_ylim(-3.3, 3.3); ax.set_xlim(lo, 0.12 * abs(lo))
    ax.set_xlabel("$s$"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("$h(s)$")
fig.tight_layout()
fig.savefig("bands.pdf")
