"""Figure 3: skewness S and excess kurtosis K of random three-state rings,
compared with the lower envelope of Theorem 3."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 6  # series order

def mul(a, b): return np.convolve(a, b)[:N]
def inv(a):
    r = np.zeros(N); r[0] = 1 / a[0]
    for k in range(1, N): r[k] = -np.dot(a[1:k + 1], r[k - 1::-1][:k]) / a[0]
    return r
def sqrt(a):
    r = np.zeros(N); r[0] = np.sqrt(a[0])
    for k in range(1, N): r[k] = (a[k] - np.dot(r[1:k], r[k - 1:0:-1])) / (2 * r[0])
    return r

def S_K(p, q):
    n = len(p)
    Qm = np.zeros((n, n))
    for i in range(n):
        Qm[i, (i + 1) % n] += p[i]; Qm[i, (i - 1) % n] += q[i]; Qm[i, i] -= p[i] + q[i]
    P, M = np.prod(p), np.prod(q); F = np.log(P / M)
    c = np.poly(Qm)[::-1] / (2 * np.sqrt(P * M)); c[0] += np.cosh(F / 2)
    h = np.zeros(N); h[:len(c)] = c[:N]
    dh = np.zeros(N); dh[:N - 1] = h[1:] * np.arange(1, N)
    g1 = mul(dh, inv(sqrt(mul(h, h) - np.eye(1, N)[0])))   # gamma'(s)
    kap = [None] + [(-1) ** j * g1[j] * np.prod(range(1, j + 1)) for j in range(N - 1)]
    # kap[j+1] = (-1)^j gamma^{(j+1)}(0) = kappa_{j+1}(tau)
    k2, k3, k4 = kap[2], kap[3], kap[4]
    return F, np.sqrt(F) * k3 / k2 ** 1.5, F * k4 / k2 ** 2

def f(x): return 2 * np.sqrt(x) * (1 + 4 * np.exp(x) + np.exp(2 * x)) / np.sqrt((np.exp(x) - 1) * (1 + np.exp(x)) ** 3)
def g(x): return 6 * x * (1 + 8 * np.exp(x) + np.exp(2 * x)) / (np.exp(2 * x) - 1)

rng = np.random.default_rng(1)
pts = []
while len(pts) < 4000:
    p, q = 10 ** rng.uniform(-3, 3, 3), 10 ** rng.uniform(-3, 3, 3)
    if np.prod(p) < np.prod(q): p, q = q, p
    F, S, K = S_K(p, q)
    if 0.05 < F < 12: pts.append((F, S, K))
pts = np.array(pts)
Fs = np.linspace(0.02, 12, 600)
envS = np.min([f(Fs / m) for m in (1, 2, 3)], axis=0)
envK = np.min([g(Fs / m) for m in (1, 2, 3)], axis=0)
rS = pts[:, 1] / np.min([f(pts[:, 0] / m) for m in (1, 2, 3)], axis=0)
rK = pts[:, 2] / np.min([g(pts[:, 0] / m) for m in (1, 2, 3)], axis=0)
assert rS.min() > 1 - 1e-5 and rK.min() > 1 - 1e-5  # double-precision tolerance

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, col, env, lab, top in [(axes[0], 1, envS, "$S$", 7.5), (axes[1], 2, envK, "$K$", 75)]:
    ax.scatter(pts[:, 0], pts[:, col], s=2, color="0.6", lw=0, rasterized=True)
    ax.plot(Fs, env, color="C3", lw=1.8)
    ax.set_xlim(0, 12); ax.set_ylim(env.min() * 0.9, top)
    ax.set_xlabel("$F$"); ax.set_ylabel(lab)
fig.tight_layout()
fig.savefig("random.pdf", dpi=300)
print("points:", len(pts), "min S/bound:", rS.min(), "min K/bound:", rK.min())
