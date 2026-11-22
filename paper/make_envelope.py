"""Figure 2: lower envelopes of S and K for n = 1, 2, 3, 5, 10 states."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def f(x): return 2*np.sqrt(x)*(1+4*np.exp(x)+np.exp(2*x))/np.sqrt((np.exp(x)-1)*(1+np.exp(x))**3)
def g(x): return 6*x*(1+8*np.exp(x)+np.exp(2*x))/(np.exp(2*x)-1)
F = np.linspace(0.02, 30, 3000)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
colors = ["#1b4f8a", "#2a7f62", "#b5651d", "#8a1b5c", "#444444"]
for ax, fn, lab, absmin in [(axes[0], f, "$S$", 3.909729), (axes[1], g, "$K$", 25.09134)]:
    for m in range(1, 11):
        ax.plot(F, fn(F/m), color="0.82", lw=0.6, zorder=1)
    for n, col in zip([1, 2, 3, 5, 10], colors):
        env = np.min([fn(F/m) for m in range(1, n+1)], axis=0)
        ax.plot(F, env, color=col, lw=1.4, label=f"$n={n}$", zorder=3)
    ax.axhline(absmin, color="k", ls="--", lw=0.8, zorder=2)
    ax.set_xlabel("affinity $F$"); ax.set_ylabel("lower bound on " + lab)
    ax.set_xlim(0, 30)
axes[0].set_ylim(3.8, 4.35); axes[1].set_ylim(24.5, 30.5)
axes[1].legend(frameon=False, loc="upper right", ncol=1)
fig.tight_layout()
fig.savefig("envelope.pdf")
