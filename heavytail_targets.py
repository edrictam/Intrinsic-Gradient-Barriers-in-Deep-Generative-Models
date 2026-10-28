"""Heavy-tailed 1-d target families and exact transport-energy machinery.

Families: Student-t (df alpha), Pareto (shape alpha, scale lam), Frechet
(shape alpha, scale s).  For the increasing rearrangement T = F_target^{-1} o
F_src from a 1-d source, this module computes the transport energy

    E_src[(T')^2]                       (the per-coordinate Jacobian energy)

by three mutually independent numerical routes:

  Route Q (primary):  int (T'(z))^2 f_src(z) dz in source space, fully in
                      log-space;
  Route Y:            int I(F(y))^2 / f(y) dy in target space (profile
                      identity; Gaussian source), meshed independently;
  Route H:            orthonormal-Hermite expansion, E[(T')^2] = sum k a_k^2
                      (Gaussian source; reliable for moderate tail indices).

Near tail index 2 the integrands decay at rate ~ (alpha-2)/(2 alpha) * z^2,
so integration ranges are alpha-dependent and quantiles reach exp(1600):
everything stays in log-space end to end.  scipy's t.isf is unreliable below
q ~ 1e-150 (verified: garbage at 1e-250), and t quantiles themselves overflow
float64 near index 2, so Student-t survival/quantile functions are computed
here via the regularized-incomplete-beta hypergeometric series

    sf_t(y) = (1/2) I_x(a/2, 1/2),  x = a/(a+y^2),
    I_x(p, q) = x^p (1-x)^q / (p B(p, q)) * 2F1(p+q, 1; p+1; x),

(geometric convergence for x <= 0.45, machine-precision-verified against
scipy in the overlap) and quantiles by Newton inversion in u = log y.
"""

import numpy as np
from scipy import special, stats

from isoperimetry import Iso, PHI, phi, norm
from quadrature import gl_nodes, gl_integrate_log, assert_tail_negligible

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)
LOG_HALF = np.log(0.5)


# ----------------------------------------------------------------------------
# log of the Gaussian isoperimetric profile from log(a), for a far below
# float range: solves phi(t)/t * (1 - t^-2 + 3 t^-4 - 15 t^-6) = a for
# t = |Phi^{-1}(a)| by Newton iteration, then returns log phi(t).
# ----------------------------------------------------------------------------

def log_Iso_from_log_a(la):
    """log I(a) given la = log a; intended for a <= 1e-40 (relative accuracy
    better than 1e-8 there, improving rapidly as a decreases); vectorized."""
    la = np.asarray(la, dtype=float)
    t = np.sqrt(np.maximum(-2.0 * (la + LOG_SQRT_2PI), 1e-12))
    for _ in range(6):
        ser = 1.0 - 1.0 / t**2 + 3.0 / t**4 - 15.0 / t**6 + 105.0 / t**8
        F = -0.5 * t**2 - LOG_SQRT_2PI - np.log(t) + np.log(ser) - la
        dser = 2.0 / t**3 - 12.0 / t**5 + 90.0 / t**7 - 840.0 / t**9
        dF = -t - 1.0 / t + dser / ser
        t = t - F / dF
    out = -0.5 * t**2 - LOG_SQRT_2PI
    return out if out.ndim else float(out)


def log_Iso_safe(log_small_side):
    """log I(a) from la = log(min(a, 1-a)): exact profile when representable,
    asymptotic solver far below float range."""
    la = np.asarray(log_small_side, dtype=float)
    out = np.empty_like(la)
    hi = la > -600.0
    if np.any(hi):
        out[hi] = np.log(Iso(np.exp(la[hi])))
    if np.any(~hi):
        out[~hi] = log_Iso_from_log_a(la[~hi])
    return out if out.ndim else float(out)


# ----------------------------------------------------------------------------
# Sources (log-pdf / log-sf / log-cdf), used by Route Q
# ----------------------------------------------------------------------------

class GaussSrc:
    name = "gauss"
    logpdf = staticmethod(norm.logpdf)
    logsf = staticmethod(norm.logsf)
    logcdf = staticmethod(special.log_ndtr)


class LaplaceSrc:
    """Laplace(0, b); b = 1/sqrt(2) gives unit variance (isotropic)."""
    name = "laplace"

    def __init__(self, b=1.0 / np.sqrt(2.0)):
        self.b = float(b)

    def logpdf(self, x):
        return -np.log(2 * self.b) - np.abs(np.asarray(x, float)) / self.b

    def logsf(self, x):
        x = np.asarray(x, float)
        out = np.where(x >= 0,
                       LOG_HALF - x / self.b,
                       np.log1p(-0.5 * np.exp(np.minimum(x, 0.0) / self.b)))
        return out if out.ndim else float(out)

    def logcdf(self, x):
        return self.logsf(-np.asarray(x, float))


# ----------------------------------------------------------------------------
# Student-t: own log-sf / log-pdf / log-quantile, valid for all magnitudes
# ----------------------------------------------------------------------------

class StudentT:
    name, label = "t", "Student-$t$"
    param_tex = r"degrees of freedom $\alpha$"
    X_SERIES = 0.45           # use the 2F1 series when x = a/(a+y^2) <= this

    @staticmethod
    def var(a):
        return a / (a - 2.0) if a > 2 else np.inf

    @staticmethod
    def log_C(a):
        """log normalizer of the t density."""
        return (special.gammaln((a + 1) / 2) - special.gammaln(a / 2)
                - 0.5 * np.log(a * np.pi))

    @classmethod
    def log_c(cls, a):
        """Tail constant: sf(y) ~ c y^{-a},  c = C_a a^{(a-1)/2}."""
        return cls.log_C(a) + (a - 1.0) / 2.0 * np.log(a)

    @classmethod
    def log_f_from_logy(cls, u, a):
        """log f_t(e^u) for u = log y (any magnitude)."""
        u = np.asarray(u, float)
        return cls.log_C(a) - (a + 1.0) / 2.0 * np.logaddexp(0.0, 2.0 * u - np.log(a))

    @classmethod
    def _log_sf_series_from_logy(cls, u, a):
        """Series branch, valid where x = a/(a+y^2) <= X_SERIES."""
        u = np.asarray(u, float)
        log_x = np.log(a) - np.logaddexp(np.log(a), 2.0 * u)   # log(a/(a+y^2))
        x = np.exp(log_x)
        p, q = a / 2.0, 0.5
        S = np.ones_like(x)
        c = np.ones_like(x)
        for n in range(400):
            c = c * ((p + q + n) / (p + 1 + n)) * x
            S = S + c
            if np.max(c) < 1e-18:
                break
        return (LOG_HALF + p * log_x + q * np.log1p(-x)
                - np.log(p) - special.betaln(p, q) + np.log(S))

    @classmethod
    def log_sf_from_logy(cls, u, a):
        """log sf_t(e^u): series in the tail, scipy in the bulk (y >= 0)."""
        u = np.asarray(u, float)
        out = np.empty_like(u)
        # x <= X_SERIES  <=>  y^2 >= a (1-X)/X
        u_seam = 0.5 * np.log(a * (1 - cls.X_SERIES) / cls.X_SERIES)
        tail = u >= u_seam
        if np.any(tail):
            out[tail] = cls._log_sf_series_from_logy(u[tail], a)
        if np.any(~tail):
            out[~tail] = stats.t.logsf(np.exp(u[~tail]), a)
        return out if out.ndim else float(out)

    @classmethod
    def log_quantile_from_lsf(cls, lsf, a):
        """u = log y solving log sf_t(e^u) = lsf, for lsf <= log(1/2);
        scipy in the bulk, log-space Newton on the series branch."""
        lsf = np.asarray(lsf, float)
        out = np.empty_like(lsf)
        u_seam = 0.5 * np.log(a * 1.5)                 # x = 0.4: safely in series zone
        lsf_seam = float(cls.log_sf_from_logy(np.array([u_seam]), a)[0])
        bulk = lsf >= lsf_seam
        if np.any(bulk):
            out[bulk] = np.log(np.maximum(stats.t.isf(np.exp(lsf[bulk]), a), 1e-300))
        if np.any(~bulk):
            lv = lsf[~bulk]
            u = (cls.log_c(a) - lv) / a                # asymptotic init
            for _ in range(6):
                F = cls._log_sf_series_from_logy(u, a) - lv
                dF = -np.exp(u + cls.log_f_from_logy(u, a)
                             - cls._log_sf_series_from_logy(u, a))
                u = u - F / dF
            out[~bulk] = u
        return out if out.ndim else float(out)

    # ---- transport quantities ----------------------------------------------

    @classmethod
    def T_gauss(cls, z, a, cap=700.0):
        """Quantile map for the standard-normal source (linear space, capped
        at exp(cap) to avoid inf*0 in downstream weighted sums)."""
        z = np.asarray(z, float)
        u = cls.log_quantile_from_lsf(norm.logsf(np.abs(z)), a)
        out = np.sign(z) * np.exp(np.minimum(u, cap))
        return out if out.ndim else float(out)

    @classmethod
    def logTprime(cls, z, a, src=GaussSrc):
        """log T'(x) = log f_src(x) - log f_t(T(x)) for a symmetric source."""
        z = np.asarray(z, float)
        zz = np.abs(z)
        u = cls.log_quantile_from_lsf(src.logsf(zz), a)
        out = src.logpdf(zz) - cls.log_f_from_logy(u, a)
        return out if out.ndim else float(out)

    @classmethod
    def E_T1prime2(cls, a, src=GaussSrc, n=3000):
        """Route Q: E[(T')^2] = 2 int_0^inf exp(2 log T' + log f_src) dx."""
        rate = (a - 2.0) / (2.0 * a)
        if isinstance(src, LaplaceSrc):
            xmax = src.b * max(120.0, 100.0 * a / (a - 2.0))
            pts = [0, src.b, 5 * src.b, xmax / 8, xmax / 4, xmax / 2, xmax]
        else:
            xmax = min(200.0, np.sqrt(90.0 / max(rate, 1e-12)))
            pts = [0, 1, 3, 8, xmax / 4, xmax / 2, 3 * xmax / 4, xmax]
        pts = sorted(set(min(p, pts[-1]) for p in pts))
        logf = lambda x: 2.0 * cls.logTprime(x, a, src) + src.logpdf(x)
        total, segs = gl_integrate_log(logf, pts, n)
        assert_tail_negligible(segs, 1e-10, f"E[(T')^2] t(alpha={a}) {src.name if not isinstance(src, type) else src.name}")
        return 2.0 * total

    @classmethod
    def E_T1prime2_yroute(cls, a, n=3000):
        """Route Y: E[(T')^2] = 2 int_0^inf I(min(F,Fbar))^2 / f dy, meshed
        in y ([0, y_c]) then u = log y; Gaussian source; own log-sf/log-f."""
        y_c = 10.0

        def integrand_lin(y):
            Fbar = stats.t.sf(y, a)
            small = np.minimum(Fbar, 1.0 - Fbar)
            return Iso(small) ** 2 / stats.t.pdf(y, a)

        tot_lin = 0.0
        for lo, hi in [(0.0, 1.0), (1.0, y_c)]:
            xs, ws = gl_nodes(lo, hi, n)
            tot_lin += float(np.sum(ws * integrand_lin(xs)))

        Umax = max(30.0, 40.0 / (a - 2.0) + np.log(y_c) + 10.0)

        def log_integrand_u(u):
            lsf = cls.log_sf_from_logy(u, a)
            lf = cls.log_f_from_logy(u, a)
            return 2.0 * log_Iso_safe(lsf) - lf + u

        pts = sorted(set([np.log(y_c), np.log(y_c) + 3, Umax / 4, Umax / 2,
                          3 * Umax / 4, Umax]))
        tot_u, segs = gl_integrate_log(log_integrand_u, pts, n)
        assert_tail_negligible(segs, 1e-9, f"yroute t(alpha={a})")
        return 2.0 * (tot_lin + tot_u)


# ----------------------------------------------------------------------------
# Pareto: exact power-law log-sf, everything closed form
# ----------------------------------------------------------------------------

class Pareto:
    name, label = "pareto", "Pareto"
    param_tex = r"shape $\alpha$"

    def __init__(self, lam=1.0):
        self.lam = float(lam)

    def var(self, a):
        if a <= 2:
            return np.inf
        return self.lam ** 2 * a / ((a - 1.0) ** 2 * (a - 2.0))

    def T_gauss(self, z, a, cap=700.0):
        """T(z) = lam * exp(-logsf(z)/a)   (support [lam, inf))."""
        lT = np.log(self.lam) - norm.logsf(np.asarray(z, float)) / a
        return np.exp(np.minimum(lT, cap))

    def logTprime(self, z, a, src=GaussSrc):
        """log T' = log(lam/a) + log f_src - (1 + 1/a) log sf_src  — exact
        closed form for any source with exact logsf."""
        z = np.asarray(z, float)
        return (np.log(self.lam / a) + src.logpdf(z)
                - (1.0 + 1.0 / a) * src.logsf(z))

    def E_T1prime2(self, a, src=GaussSrc, n=3000):
        rate = (a - 2.0) / (2.0 * a)
        if isinstance(src, LaplaceSrc):
            xmax = src.b * max(120.0, 100.0 * a / (a - 2.0))
            pts = [-12 * src.b, 0, src.b, 5 * src.b, xmax / 4, xmax / 2, xmax]
        else:
            xmax = min(200.0, np.sqrt(90.0 / max(rate, 1e-12)))
            pts = [-10, 0, 2, 8, xmax / 4, xmax / 2, 3 * xmax / 4, xmax]
        pts = sorted(set(min(p, pts[-1]) for p in pts))
        logf = lambda x: 2.0 * self.logTprime(x, a, src) + src.logpdf(x)
        total, segs = gl_integrate_log(logf, pts, n)
        assert_tail_negligible(segs, 1e-10, f"E[(T')^2] pareto(alpha={a})")
        return total

    def E_T1prime2_yroute(self, a, n=3000):
        """Route Y over u = log(y/lam) in (0, inf): log Fbar = -a u exactly."""
        lam = self.lam
        Umax = max(30.0, 40.0 / (a - 2.0) + 10.0)
        med = np.log(2.0) / a

        def log_integrand_u(u):
            lFbar = -a * u
            lF = np.log(-np.expm1(-a * np.maximum(u, 1e-300)))
            lsmall = np.minimum(lF, lFbar)
            lf = np.log(a / lam) - (a + 1.0) * u        # log f(y)
            return 2.0 * log_Iso_safe(lsmall) - lf + np.log(lam) + u

        pts = sorted(set([1e-12, med / 4, med, med * 4, 3.0, Umax / 4,
                          Umax / 2, 3 * Umax / 4, Umax]))
        total, segs = gl_integrate_log(log_integrand_u, pts, n)
        assert_tail_negligible(segs, 1e-9, f"yroute pareto(alpha={a})")
        return total


# ----------------------------------------------------------------------------
# Frechet: exact double-exponential log-cdf, log(-log Phi) branch switch
# ----------------------------------------------------------------------------

class Frechet:
    name, label = "frechet", "Fr\\'echet"
    param_tex = r"shape $\alpha$"
    Z_SWITCH = 5.0

    def __init__(self, s=1.0):
        self.s = float(s)

    def var(self, a):
        if a <= 2:
            return np.inf
        g1 = special.gamma(1.0 - 1.0 / a)
        g2 = special.gamma(1.0 - 2.0 / a)
        return self.s ** 2 * (g2 - g1 ** 2)

    @classmethod
    def _log_neg_log_cdf_src(cls, z, src=GaussSrc):
        """log(-log F_src(z)), stable across the whole line."""
        z = np.asarray(z, float)
        out = np.empty_like(z)
        lo = z < cls.Z_SWITCH
        if np.any(lo):
            out[lo] = np.log(-src.logcdf(z[lo]))
        if np.any(~lo):
            # -log F = -log(1 - sf) = sf (1 + sf/2 + ...) ~ sf
            out[~lo] = src.logsf(z[~lo])
        return out if out.ndim else float(out)

    def T_gauss(self, z, a, cap=700.0):
        """T(z) = s * (-log Phi(z))^{-1/a}."""
        L = self._log_neg_log_cdf_src(z)
        return np.exp(np.minimum(np.log(self.s) - L / a, cap))

    def logTprime(self, z, a, src=GaussSrc):
        """log T' = log(s/a) - (1+1/a) log(-log F_src) + log f_src - log F_src."""
        z = np.asarray(z, float)
        L = self._log_neg_log_cdf_src(z, src)
        return (np.log(self.s / a) - (1.0 + 1.0 / a) * L
                + src.logpdf(z) - src.logcdf(z))

    def E_T1prime2(self, a, src=GaussSrc, n=3000):
        rate = (a - 2.0) / (2.0 * a)
        if isinstance(src, LaplaceSrc):
            xmax = src.b * max(120.0, 100.0 * a / (a - 2.0))
            pts = [-12 * src.b, 0, src.b, 5 * src.b, xmax / 4, xmax / 2, xmax]
        else:
            xmax = min(200.0, np.sqrt(90.0 / max(rate, 1e-12)))
            pts = [-10, 0, 2, 8, xmax / 4, xmax / 2, 3 * xmax / 4, xmax]
        pts = sorted(set(min(p, pts[-1]) for p in pts))
        logf = lambda x: 2.0 * self.logTprime(x, a, src) + src.logpdf(x)
        total, segs = gl_integrate_log(logf, pts, n)
        assert_tail_negligible(segs, 1e-10, f"E[(T')^2] frechet(alpha={a})")
        return total

    def E_T1prime2_yroute(self, a, n=3000):
        """Route Y over u = log(y/s): F = exp(-e^{-a u}); expm1 branch for the
        survival side, direct log F on the left tail."""
        s = self.s
        Umax = max(30.0, 40.0 / (a - 2.0) + 10.0)
        Umin = -6.5 / a
        med = -np.log(np.log(2.0)) / a

        def log_integrand_u(u):
            w = np.exp(-a * u)                 # (y/s)^{-a}
            lF = -w
            with np.errstate(divide="ignore"):
                lFbar = np.where(w > 1e-300, np.log(-np.expm1(-w)), -a * u)
            lsmall = np.minimum(lF, lFbar)
            lf = np.log(a / s) - (a + 1.0) * u - w
            return 2.0 * log_Iso_safe(lsmall) - lf + np.log(s) + u

        pts = sorted(set([Umin, med - 1.0, med, med + 1.0, 3.0,
                          Umax / 4, Umax / 2, 3 * Umax / 4, Umax]))
        total, segs = gl_integrate_log(log_integrand_u, pts, n)
        assert_tail_negligible(segs, 1e-9, f"yroute frechet(alpha={a})")
        return total


# ----------------------------------------------------------------------------
# Route H: orthonormal Hermite expansion (Gaussian source), moderate indices
# ----------------------------------------------------------------------------

def hermite_energy(T_of_z, K=60, n_nodes=400):
    """Return (Var_H, E[(T')^2]_H) from the orthonormal-Hermite coefficients
    a_k = E[T(Z) h_k(Z)]:  Var = sum_{k>=1} a_k^2,  E[(T')^2] = sum k a_k^2.

    h_0 = 1, h_1 = z, h_{k+1} = (z h_k - sqrt(k) h_{k-1}) / sqrt(k+1).
    Zero-weight extreme nodes are dropped (they underflow the Gaussian weight
    long before T explodes, but 0 * inf must never be formed).
    """
    z, w = special.roots_hermitenorm(n_nodes)    # weight e^{-z^2/2}
    w = w / np.sqrt(2 * np.pi)
    keep = w > 0.0
    z, w = z[keep], w[keep]
    Tz = T_of_z(z)
    h_prev = np.ones_like(z)
    h_cur = z.copy()
    coeffs = [float(np.sum(w * Tz * h_prev)), float(np.sum(w * Tz * h_cur))]
    for k in range(1, K):
        h_next = (z * h_cur - np.sqrt(k) * h_prev) / np.sqrt(k + 1.0)
        h_prev, h_cur = h_cur, h_next
        coeffs.append(float(np.sum(w * Tz * h_cur)))
    a = np.asarray(coeffs)
    var_H = float(np.sum(a[1:] ** 2))
    energy_H = float(np.sum(np.arange(1, len(a)) * a[1:] ** 2))
    return var_H, energy_H


FAMILIES = {
    "t": StudentT,
    "pareto": Pareto(lam=1.0),
    "frechet": Frechet(s=1.0),
}


def param_grid(name, quick=False):
    if name == "t":
        g = np.concatenate([np.arange(2.05, 30.0001, 0.25),
                            np.arange(35.0, 120.0001, 5.0)])
    else:
        g = np.concatenate([np.arange(2.05, 20.0001, 0.25),
                            np.arange(22.5, 80.0001, 2.5)])
    return g[::8] if quick else g
