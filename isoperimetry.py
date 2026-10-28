"""Isoperimetric-type profiles of 1-d source (latent) distributions.

The Gaussian isoperimetric profile I(a) = phi(Phi^{-1}(a)) is the central
object of the multimodal half of the paper.  For a general 1-d source with
density f_src and CDF F_src, the analogous "transport profile" is

    I_src(a) = f_src(F_src^{-1}(a)),      a in (0, 1),

which enters through the exact identity (Prop. \\ref{prop:identity} and its
generalizations): for the increasing rearrangement g = F_nu^{-1} o F_src,

    E_src[g'] = int_R I_src(F_nu(y)) dy.

Profiles implemented here:
  * gaussian:          I(a) = phi(Phi^{-1}(a))                  (standard normal)
  * gaussian_kappa:    sqrt(kappa) * I(a)                       (N(0, 1/kappa))
  * laplace:           min(a, 1-a) / b                          (Laplace(0, b))
"""

import numpy as np
from scipy import optimize, stats

norm = stats.norm
PHI, phi = norm.cdf, norm.pdf

I_HALF = phi(0.0)  # = (2*pi)^(-1/2)


def Iso(a):
    """I(a) = phi(Phi^{-1}(a)), extended by 0 outside (0,1) (and for a<0)."""
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    inside = (a > 0.0) & (a < 1.0)
    small = inside & (np.minimum(a, 1 - a) < 1e-300)
    reg = inside & ~small
    out[reg] = phi(norm.ppf(a[reg]))
    if np.any(small):
        b = np.minimum(a[small], 1 - a[small])
        out[small] = b * np.sqrt(-2 * np.log(b))   # -log avoids 1/b overflow
    return out if out.ndim else float(out)


def IsoInv(c):
    """Capped inverse: sup{a in [0,1/2] : I(a) <= c}; = 1/2 for c >= I(1/2)."""
    c = float(c)
    if c <= 0.0:
        return 0.0
    if c >= I_HALF:
        return 0.5
    return optimize.brentq(lambda a: Iso(a) - c, 1e-300, 0.5, xtol=1e-15, rtol=8.9e-16)


def profile_gaussian(a):
    """Transport profile of the standard normal source (= Iso)."""
    return Iso(a)


def make_profile_gaussian_kappa(kappa):
    """Transport profile of N(0, 1/kappa): sqrt(kappa) * Iso(a).

    (Density of N(0, s^2) at its a-quantile is phi(Phi^{-1}(a))/s; here
    s = kappa^{-1/2}.)  Matches the sqrt(kappa) gain of Thm \\ref{thm:strongly}.
    """
    root_kappa = float(np.sqrt(kappa))

    def profile(a):
        return root_kappa * Iso(a)

    return profile


def make_profile_laplace(b):
    """Transport profile of Laplace(0, b): min(a, 1-a)/b.

    (For a <= 1/2 the a-quantile is b*log(2a) with density a/b; symmetric on
    the other side.)  The Cheeger constant of Laplace(0, b) is h = 1/b, and
    min(a, 1-a)/b = h * min(a, 1-a) matches Thm \\ref{thm:cheeger}.
    """
    b = float(b)

    def profile(a):
        a = np.asarray(a, dtype=float)
        out = np.minimum(np.clip(a, 0.0, 1.0), np.clip(1 - a, 0.0, 1.0)) / b
        return out if out.ndim else float(out)

    return profile


class GaussianSource:
    """Standard normal source: pdf/cdf/ppf + transport profile."""

    profile = staticmethod(Iso)
    pdf = staticmethod(phi)
    cdf = staticmethod(PHI)
    ppf = staticmethod(norm.ppf)
    logpdf = staticmethod(norm.logpdf)


class LaplaceSource:
    """Laplace(0, b) source: pdf/cdf/ppf + transport profile min(a,1-a)/b."""

    def __init__(self, b):
        self.b = float(b)
        self._dist = stats.laplace(scale=self.b)
        self.profile = make_profile_laplace(self.b)

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, p):
        return self._dist.ppf(p)

    def logpdf(self, x):
        return self._dist.logpdf(x)
