"""1-d mixture targets and exact transport machinery.

`Target1D` represents nu = sum_i w_i N(mu_i, s_i^2) (+ optional uniform valley
fill) and exposes the increasing rearrangement g = F_nu^{-1} o F_src from a
1-d source (standard normal by default), together with exact quadrature
routes for E_src[g'] and its valley localizations:

  route (y):  E_src[g'] = int_R I_src(F_nu(y)) dy      (profile identity)
  route (x):  E_src[g'] = int f_src(x)^2 / f_nu(g(x)) dx

where I_src(a) = f_src(F_src^{-1}(a)) is the source's transport profile
(see isoperimetry).  Both routes are deterministic Gauss-Legendre.
"""

from functools import lru_cache

import numpy as np
from scipy import optimize, special, stats

from isoperimetry import Iso, PHI, phi, norm

__all__ = ["Target1D", "two_gauss"]


class Target1D:
    """nu = sum_i w_i N(mu_i, s_i^2)  (+ optional delta*Unif[-D/2, D/2])."""

    def __init__(self, weights, mus, sigmas, unif_mass=0.0, unif_halfwidth=None):
        self.w = np.asarray(weights, float)
        self.mu = np.asarray(mus, float)
        self.s = np.asarray(sigmas, float)
        self.u_mass = float(unif_mass)
        self.u_hw = unif_halfwidth
        assert abs(self.w.sum() + self.u_mass - 1.0) < 1e-12

    def cdf(self, y):
        y = np.asarray(y, float)
        out = sum(w * PHI((y - m) / s) for w, m, s in zip(self.w, self.mu, self.s))
        if self.u_mass:
            hw = self.u_hw
            out = out + self.u_mass * np.clip((y + hw) / (2 * hw), 0.0, 1.0)
        return out

    def pdf(self, y):
        y = np.asarray(y, float)
        out = sum(w * phi((y - m) / s) / s for w, m, s in zip(self.w, self.mu, self.s))
        if self.u_mass:
            hw = self.u_hw
            out = out + np.where(np.abs(y) <= hw, self.u_mass / (2 * hw), 0.0)
        return out

    def var(self):
        """Exact variance of the mixture (components + optional uniform)."""
        mean = float(np.sum(self.w * self.mu))
        second = float(np.sum(self.w * (self.s ** 2 + self.mu ** 2)))
        if self.u_mass:
            second += self.u_mass * (self.u_hw ** 2) / 3.0
        return second - mean ** 2

    def bracket(self):
        lo = float(np.min(self.mu - 60 * self.s)) - 10.0
        hi = float(np.max(self.mu + 60 * self.s)) + 10.0
        if self.u_mass:
            lo, hi = min(lo, -self.u_hw - 10), max(hi, self.u_hw + 10)
        return lo, hi

    def quantile(self, p):
        lo, hi = self.bracket()
        return optimize.brentq(lambda y: self.cdf(y) - p, lo, hi,
                               xtol=1e-13, rtol=8.9e-16)

    def quantile_vec(self, p, iters=64):
        """Vectorized bisection for F^{-1}(p) on arrays (deterministic)."""
        p = np.asarray(p, float)
        lo, hi = self.bracket()
        lo = np.full_like(p, lo)
        hi = np.full_like(p, hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    # transport g = F^{-1} o F_src and derived quantities ---------------------
    def g_of_x(self, x, source=None):
        Fs = PHI if source is None else source.cdf
        return self.quantile(Fs(x))

    @staticmethod
    @lru_cache(maxsize=8)
    def _gl_raw(n):
        return special.roots_legendre(n)

    def _gl_nodes(self, a, b, n):
        xs, ws = self._gl_raw(n)
        return (a + b) / 2 + (b - a) / 2 * xs, (b - a) / 2 * ws

    def mean_gprime_via_profile(self, breakpoints=(), n_nodes=2000, profile=Iso):
        """E_src[g'] = int_R I_src(F(y)) dy  (route y); vectorized GL per
        segment.  `profile` is the source's transport profile (default:
        Gaussian isoperimetric profile)."""
        lo, hi = self.bracket()
        pts = sorted(set([lo, hi] + list(map(float, breakpoints))
                         + list(self.mu) + list(self.mu - 9 * self.s)
                         + list(self.mu + 9 * self.s)))
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            ys, ws = self._gl_nodes(a, b, n_nodes)
            total += float(np.sum(ws * profile(self.cdf(ys))))
        return total

    def mean_gprime_via_x(self, yL=None, yR=None, xmax=9.0, source=None):
        """E_src[g'] = int g'(x) f_src(x) dx with g'(x)=f_src(x)/f_nu(g(x))
        (route x).  Gauss-Legendre + vectorized bisection, independent of
        route (y).  The integral is split at the valley preimage (yL, yR) so
        the transport-derivative peak gets its own densely resolved segment.
        """
        f_src = phi if source is None else source.pdf
        F_src = PHI if source is None else source.cdf
        Q_src = norm.ppf if source is None else source.ppf
        cuts = [-xmax, xmax]
        if yL is not None:
            cuts += [Q_src(self.cdf(yL)), Q_src(self.cdf(yR))]
        cuts = sorted(cuts)
        total = 0.0
        for a, b, n in zip(cuts[:-1], cuts[1:], (2000, 4000, 2000)):
            xs, ws = self._gl_nodes(a, b, n)
            vals = f_src(xs) ** 2 / self.pdf(self.quantile_vec(F_src(xs)))
            total += float(np.sum(ws * vals))
        return total

    def shell_integral_x(self, yL, yR, n_nodes=2000, source=None):
        """int_{g^{-1}((yL,yR))} g' dF_src computed natively in x-space."""
        f_src = phi if source is None else source.pdf
        F_src = PHI if source is None else source.cdf
        Q_src = norm.ppf if source is None else source.ppf
        xL = Q_src(self.cdf(yL))
        xR = Q_src(self.cdf(yR))
        xs, ws = self._gl_nodes(xL, xR, n_nodes)
        vals = f_src(xs) ** 2 / self.pdf(self.quantile_vec(F_src(xs)))
        return float(np.sum(ws * vals))

    def profile_integral_y(self, yL, yR, n_nodes=4000, profile=Iso):
        """S = int_{yL}^{yR} I_src(F(y)) dy."""
        ys, ws = self._gl_nodes(yL, yR, n_nodes)
        return float(np.sum(ws * profile(self.cdf(ys))))

    def gprime_on_grid(self, ygrid, profile=Iso):
        """g' as a function of y: I_src(F(y)) / f_nu(y)  (exact, pointwise)."""
        return profile(self.cdf(ygrid)) / self.pdf(ygrid)


def two_gauss(Delta, sigma, offset=3.0):
    """Symmetric two-Gaussian target with interval width Delta between the
    3-sigma edges: modes at +-(Delta/2 + offset*sigma)."""
    a = Delta / 2 + offset * sigma
    return Target1D([0.5, 0.5], [-a, a], [sigma, sigma]), a
