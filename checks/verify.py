"""Numerical checks for "Sharp lower bounds on first-passage skewness and
kurtosis in unicyclic networks".  Requires mpmath.  Run: python verify.py"""
import random
import mpmath as mp

mp.mp.dps = 40
KMAX = 6


def ring_poly(p, q):
    """Return (F, h): the affinity and the spectral polynomial h(s) of the ring."""
    n = len(p)
    Q = mp.zeros(n, n)
    for i in range(n):
        Q[i, (i + 1) % n] += p[i]
        Q[i, (i - 1) % n] += q[i]
        Q[i, i] -= p[i] + q[i]
    P, M = mp.fprod(p), mp.fprod(q)
    F = mp.log(P / M)
    return F, (lambda s: mp.cosh(F / 2) + mp.det(s * mp.eye(n) - Q) / (2 * mp.sqrt(P * M)))


def moments(h, F, kmax=KMAX):
    """M_j = (-1)^{j-1} gamma^{(j)}(0)/(j-1)!, normalized so that M_1 = 1."""
    d = mp.taylor(lambda s: mp.acosh(h(s)) - F / 2, 0, kmax)
    raw = [None] + [(-1) ** (j - 1) * d[j] * j for j in range(1, kmax + 1)]
    # time rescaling to unit mean: M_j -> M_j / M_1^j
    return [None] + [raw[j] / raw[1] ** j for j in range(1, kmax + 1)]


def Hk(k, j, F):
    c = mp.coth(F / (2 * j))
    return j ** (1 - mp.mpf(k) / 2) * mp.legendre(k - 1, c) / c ** (mp.mpf(k) / 2)


def f(x): return 2 * mp.sqrt(x) * (1 + 4 * mp.e ** x + mp.e ** (2 * x)) / mp.sqrt((mp.e ** x - 1) * (1 + mp.e ** x) ** 3)
def g(x): return 6 * x * (1 + 8 * mp.e ** x + mp.e ** (2 * x)) / (mp.e ** (2 * x) - 1)


def Ak(M, k, F):
    return mp.factorial(k - 1) * F ** (mp.mpf(k) / 2 - 1) * M[k] / M[2] ** (mp.mpf(k) / 2)


ok = True
def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))


# 1. Homogeneous cycles: A_k equals (k-1)! F^{k/2-1} H_k, and A_3 = f, A_4 = g.
err = 0
for n, F0 in [(1, 0.7), (2, 3.0), (3, 2.0), (5, 7.0), (7, 15.0)]:
    a = mp.e ** (F0 / n)
    F, h = ring_poly([a] * n, [mp.mpf(1)] * n)
    M = moments(h, F)
    for k in range(3, KMAX + 1):
        err = max(err, abs(Ak(M, k, F) / (mp.factorial(k - 1) * F ** (mp.mpf(k) / 2 - 1) * Hk(k, n, F)) - 1))
    err = max(err, abs(Ak(M, 3, F) / f(F / n) - 1), abs(Ak(M, 4, F) / g(F / n) - 1))
report("homogeneous formula and f, g", err < mp.mpf(10) ** -25, "max rel err %s" % mp.nstr(err, 3))

# 2. Random rings satisfy Theorem 2 for k = 3..6.
random.seed(2026)
viol, worst, N = 0, {k: mp.inf for k in range(3, KMAX + 1)}, 0
for _ in range(1500):
    n = random.randint(1, 6)
    p = [mp.mpf(10 ** random.uniform(-3, 3)) for _ in range(n)]
    q = [mp.mpf(10 ** random.uniform(-3, 3)) for _ in range(n)]
    if mp.fprod(p) < mp.fprod(q):
        p, q = q, p
    F, h = ring_poly(p, q)
    if F < 1e-3:
        continue
    M = moments(h, F)
    N += 1
    for k in range(3, KMAX + 1):
        bound = mp.factorial(k - 1) * F ** (mp.mpf(k) / 2 - 1) * min(Hk(k, j, F) for j in range(1, n + 1))
        r = Ak(M, k, F) / bound
        worst[k] = min(worst[k], r)
        viol += r < 1 - mp.mpf(10) ** -20
report("random rings obey Theorem 2, k=3..6", viol == 0,
       "%d rings, min A_k/bound: %s" % (N, ", ".join("k=%d:%s" % (k, mp.nstr(worst[k], 8)) for k in worst)))

# 3. Sharpness (sharpness lemma): block construction approaches h_m at exactly n states.
def block_ring(Ls, F, R):
    m = len(Ls); a = mp.e ** (F / m); b = mp.mpf(1)
    p, q = [], []
    for i, L in enumerate(Ls):
        for t in range(L):
            p.append(L * a if t == L - 1 else R)
            q.append(Ls[i] * b if t == 0 else R)
    return p, q
F0 = mp.mpf(4)
for Ls in [(1, 3), (2, 2, 1), (1, 1, 1, 2)]:
    m, n = len(Ls), sum(Ls)
    errs = []
    for R in [mp.mpf(10) ** 3, mp.mpf(10) ** 6]:
        F, h = ring_poly(*block_ring(Ls, F0, R))
        M = moments(h, F)
        errs.append(abs(Ak(M, 3, F) - f(F / m)) + abs(Ak(M, 4, F) - g(F / m)))
    report("contraction n=%d -> m=%d" % (n, m), abs(F - F0) < 1e-30 and errs[1] < errs[0] / 100 and errs[1] < 1e-4,
           "|A3-f|+|A4-g| at R=1e3,1e6: %s, %s" % (mp.nstr(errs[0], 3), mp.nstr(errs[1], 3)))

# 4. Quadratic arc (two-state proposition): factorization of h^2-1, moment identity, no interior minimum.
F = mp.mpf(3); T = mp.coth(F / 4); c = (T + 1 / T) / 2
def quad_h(v):
    # h = C + D s + a s^2 with variance v (unit mean): solve for a
    C, D = mp.cosh(F / 2), mp.sinh(F / 2)
    def var(a):
        return moments(lambda s: C + D * s + a * s ** 2, F, 2)[2] - v
    a = mp.findroot(var, (mp.mpf('1e-9'), D ** 2 / (4 * (C + 1))), solver='anderson')
    return (lambda s: C + D * s + a * s ** 2), a
idmax = 0
for v in [T / 2 + (c - T / 2) * x for x in (mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.8'))]:
    h, a = quad_h(v)
    qq = c - v
    Pt = lambda t: (1 - T * t + T * qq / 2 * t ** 2) * (1 - t / T + qq / (2 * T) * t ** 2)
    M = moments(h, F, 8)
    fr = mp.taylor(lambda t: Pt(t) ** mp.mpf(-0.5), 0, 8)
    for k in range(2, 9):
        idmax = max(idmax, abs(M[k] - (fr[k - 1] - qq * fr[k - 2])))
report("quadratic arc: M_k = f_{k-1} - q f_{k-2}", idmax < mp.mpf(10) ** -20, "max err %s" % mp.nstr(idmax, 3))
vs = [T / 2 + (c - T / 2) * mp.mpf(i) / 40 for i in range(1, 40)]
Qs = {k: [] for k in range(3, 9)}
for v in vs:
    M = moments(quad_h(v)[0], F, 8)
    for k in Qs:
        Qs[k].append(M[k] / M[2] ** (mp.mpf(k) / 2))
def interior_min(L):
    return any(L[i] < L[i - 1] and L[i] < L[i + 1] for i in range(1, len(L) - 1))
report("quadratic arc: no interior minimum of Q_k, k=3..8", not any(interior_min(Qs[k]) for k in Qs))

# 5. One-gap cubic arc (one-gap barrier): right gap closed, left gap open.
F = mp.mpf(5); C, D = mp.cosh(F / 2), mp.sinh(F / 2)
def h_face(r, y):
    # h' = D (1 - s/r)(1 - s/y): open critical point r, closed critical point y
    a2 = -D * (1 / r + 1 / y) / 2
    a3 = D / (3 * r * y)
    return a2, a3, (lambda s: C + D * s + a2 * s ** 2 + a3 * s ** 3)
h3 = mp.taylor(lambda s: mp.chebyt(3, mp.cosh(F / 6) + s / 3 * mp.sinh(F / 6)), 0, 3)
r0, y0 = sorted(mp.polyroots([3 * h3[3], 2 * h3[2], D]))
arc = []
for i in range(-40, 41):
    r = r0 * (1 + mp.mpf(i) / 100)
    try:
        y = mp.findroot(lambda y: h_face(r, y)[2](y) + 1, y0)
    except Exception:
        continue
    a2, a3, hh = h_face(r, y)
    if not (r < y < 0) or hh(r) <= 1 + mp.mpf(10) ** -12:
        continue  # need r leftmost and its gap open
    arc.append((a2, a3, hh, r, y))
M_arc = [moments(a[2], F, 8) for a in arc]
v_arc = [M[2] for M in M_arc]
mono = all(v_arc[i] != v_arc[i + 1] for i in range(len(v_arc) - 1))
order = sorted(range(len(arc)), key=lambda i: v_arc[i])
bad = any(interior_min([M_arc[i][k] / M_arc[i][2] ** (mp.mpf(k) / 2) for i in order]) for k in range(3, 9))
report("one-gap cubic arc: no interior minimum of Q_k, k=3..8", len(arc) > 20 and not bad, "%d arc points" % len(arc))
# quartic representation and ordering 0 < x1 < x2 < q < x3 < x4 on the arc
a2, a3, hh, x1, x2 = arc[len(arc) // 3]
roots = mp.polyroots([a3 ** 2, 2 * a2 * a3, a2 ** 2 + 2 * D * a3, 2 * C * a3 + 2 * D * a2, D ** 2 + 2 * C * a2, 2 * C * D, C ** 2 - 1], maxsteps=200, extraprec=200)
simple = sorted([r.real for r in roots if abs(r.real - x2) > mp.mpf(10) ** -8])
xs = sorted([-1 / r for r in simple]); qv = -1 / x1
Pq = lambda t: mp.fprod([1 - xi * t for xi in xs])
M = moments(hh, F, 8)
fr = mp.taylor(lambda t: Pq(t) ** mp.mpf(-0.5), 0, 8)
rep_err = max(abs(M[k] - (fr[k - 1] - qv * fr[k - 2])) for k in range(2, 9))
report("one-gap quartic representation", rep_err < mp.mpf(10) ** -15 and xs[1] < qv < xs[2],
       "err %s, ordering x2<q<x3: %s" % (mp.nstr(rep_err, 3), xs[1] < qv < xs[2]))

# 6. Selector constants and absolute minima (selection theorem, absolute bounds).
z3 = mp.findroot(lambda x: mp.diff(f, x), 2.3) / 2
z4 = mp.findroot(lambda x: mp.diff(g, x), 2.6) / 2
report("z_3, z_4, absolute minima", abs(z3 - mp.mpf('1.150671')) < 1e-6 and abs(z4 - mp.mpf('1.320669')) < 1e-6
       and abs(f(2 * z3) - mp.mpf('3.909729')) < 1e-6 and abs(g(2 * z4) - mp.mpf('25.09134')) < 1e-5,
       "z3=%s z4=%s Smin=%s Kmin=%s" % (mp.nstr(z3, 10), mp.nstr(z4, 10), mp.nstr(f(2 * z3), 10), mp.nstr(g(2 * z4), 10)))
# unimodality of H_k(j;F) in real j and switch ordering
uni = True
for k in range(3, 9):
    Fk = mp.mpf(12)
    vals = [Hk(k, mp.mpf(j) / 10, Fk) for j in range(2, 400)]
    sgn = [vals[i + 1] < vals[i] for i in range(len(vals) - 1)]
    uni &= sgn == sorted(sgn, reverse=True)  # True...True False...False
report("H_k(j;F) decreases then increases in j, k=3..8", uni)

# 7. Linear-response limit (linear-response remark): every branch -> 2^{k/2-1}(2k-3)!! as F -> 0.
lr = max(abs(mp.factorial(k - 1) * F ** (mp.mpf(k) / 2 - 1) * Hk(k, j, F) / (2 ** (mp.mpf(k) / 2 - 1) * mp.fac2(2 * k - 3)) - 1)
         for k in range(3, 9) for j in (1, 2, 5, 20) for F in [mp.mpf(10) ** -12])
report("F->0 limit 2^{k/2-1}(2k-3)!!, k=3..8 (S->3sqrt2, K->30)", lr < mp.mpf(10) ** -9, "max rel err %s" % mp.nstr(lr, 3))

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
