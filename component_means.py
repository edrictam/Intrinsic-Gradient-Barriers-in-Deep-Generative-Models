"""Generate Figure 6 for the five-component gap-compression examples."""
from pathlib import Path
import sys, os, csv, math, tempfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'annals-intrinsic-gradients-cache'))
import numpy as np
from scipy import special, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import RED, BLUE, JACOBIAN_LABEL, apply_style

SQRT2 = np.sqrt(2.)
HROOT = np.sqrt(np.pi/2.)
PATTERN = np.array([.8, -.6, 1., -.8, .4])
RADIUS, BANDWIDTH = 1.5, .4


class GapWarp:
    """Positive slope determined ONLY by rough centres, radius, bandwidth."""
    def __init__(self, centres, radius=RADIUS, bandwidth=BANDWIDTH):
        self.centres = np.sort(np.asarray(centres, dtype=float))
        self.r, self.h = float(radius), float(bandwidth)
        assert self.r >= 0 and self.h > 0
        cores = []
        for c in self.centres:
            lo, hi = c-self.r, c+self.r
            if cores and lo <= cores[-1][1]: cores[-1][1] = max(hi, cores[-1][1])
            else: cores.append([lo, hi])
        self.cores = cores
        self.left, self.right = cores[0][0], cores[-1][1]
        self.segments = []
        prefix = self.left
        for i, (lo, hi) in enumerate(cores):
            self.segments.append((lo, hi, 'core', lo, prefix)); prefix += hi-lo
            if i+1 < len(cores):
                end = cores[i+1][0]; mid = (hi+end)/2
                self.segments.append((hi, mid, 'left', hi, prefix))
                increment = self.h*HROOT*special.erf((mid-hi)/(SQRT2*self.h))
                prefix += increment
                self.segments.append((mid, end, 'right', end, prefix)); prefix += increment
        self.output_right = prefix
        self.shift = float(self._unanchored(np.array([0.]))[0])

    def log_slope(self, x):
        x = np.asarray(x, dtype=float)
        distance = np.min(np.abs(x[..., None]-self.centres), axis=-1)
        out = -.5*(np.maximum(distance-self.r, 0.)/self.h)**2
        return np.where((x < self.left) | (x > self.right), 0., out)

    def _unanchored(self, x):
        x = np.asarray(x, dtype=float); y = np.empty_like(x)
        y[x <= self.left] = x[x <= self.left]
        y[x >= self.right] = self.output_right+x[x >= self.right]-self.right
        for lo, hi, kind, edge, prefix in self.segments:
            m = (x >= lo) & (x <= hi); u = x[m]
            if kind == 'core': val = u-lo
            elif kind == 'left': val = self.h*HROOT*special.erf((u-edge)/(SQRT2*self.h))
            else:
                val = self.h*HROOT*(special.erfc((edge-u)/(SQRT2*self.h))-special.erfc((edge-lo)/(SQRT2*self.h)))
            y[m] = prefix+val
        return y

    def __call__(self, x): return self._unanchored(x)-self.shift

    def minimum_log_slope(self):
        gaps = [self.cores[j+1][0]-self.cores[j][1] for j in range(len(self.cores)-1)]
        return -max(gaps, default=0.)**2/(8*self.h*self.h)


class Mixture:
    def __init__(self, means, sigmas, weights):
        self.means, self.sigmas, self.weights = [np.asarray(v, dtype=float) for v in (means, sigmas, weights)]
        assert np.all(self.sigmas > 0) and np.all(self.weights > 0) and abs(self.weights.sum()-1) < 1e-12
        self.lw = np.log(self.weights)

    def logs(self, x):
        u = (np.asarray(x)[:, None]-self.means)/self.sigmas
        lc = special.logsumexp(self.lw+special.log_ndtr(u), axis=1)
        ls = special.logsumexp(self.lw+special.log_ndtr(-u), axis=1)
        lp = special.logsumexp(self.lw-np.log(self.sigmas)+stats.norm.logpdf(u), axis=1)
        return lc, ls, lp

    def quantile_from_gaussian(self, z):
        z = np.asarray(z); qs = self.means+self.sigmas*z[:, None]
        lo, hi = qs.min(axis=1), qs.max(axis=1)
        sign = np.where(z <= 0, 1., -1.); target = special.log_ndtr(sign*z)
        for _ in range(65):
            mid = (lo+hi)/2
            u = sign[:, None]*(mid[:, None]-self.means)/self.sigmas
            val = special.logsumexp(self.lw+special.log_ndtr(u), axis=1)
            left = np.where(sign > 0, val < target, val > target)
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
        return (lo+hi)/2

    def variance(self):
        mean = np.dot(self.weights, self.means)
        return float(np.dot(self.weights, self.sigmas**2+(self.means-mean)**2))


def nodes(cuts, n):
    q, w = np.polynomial.legendre.leggauss(n)
    cuts = np.array(sorted(set(float(v) for v in cuts)))
    lo, hi = cuts[:-1, None], cuts[1:, None]
    return ((hi+lo)/2+(hi-lo)/2*q).ravel(), ((hi-lo)/2*w).ravel()


def target_cuts(mix, warp, extent=14):
    low = mix.means.min()-extent*mix.sigmas.max()
    high = mix.means.max()+extent*mix.sigmas.max()
    cuts = [low, high, 0., *warp.centres]
    for lo, hi, _, _, _ in warp.segments: cuts += [lo, hi]
    for m, s in zip(mix.means, mix.sigmas):
        cuts += list(m+s*np.array([-18,-14,-10,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,10,14,18]))
    for i in range(len(mix.means)-1):
        m, n = mix.means[i:i+2]; s, t = mix.sigmas[i:i+2]; w, v = mix.weights[i:i+2]
        coef = [1/(2*t*t)-1/(2*s*s), m/(s*s)-n/(t*t), n*n/(2*t*t)-m*m/(2*s*s)+np.log(w*t/(v*s))]
        roots = np.roots(np.trim_zeros(np.asarray(coef), 'f'))
        cross = [float(q.real) for q in roots if abs(q.imag) < 1e-8 and m < q.real < n]
        for point in [(m+n)/2, *cross]:
            scale = min(s, t)**2/(n-m)
            cuts += list(point+scale*np.array([-32,-12,-4,-1,-.25,0,.25,1,4,12,32]))
    return sorted(set(float(v) for v in cuts if low <= v <= high))


def gaussian_log_profile(logprob):
    return stats.norm.logpdf(special.ndtri_exp(logprob))


def target_metrics(mix, warp, n, extent=14):
    x, w = nodes(target_cuts(mix, warp, extent), n)
    lc, ls, lp = mix.logs(x); li = gaussian_log_profile(np.minimum(lc, ls))
    la = warp.log_slope(x); lw = np.log(w)
    result = {}
    for name, slope in [('original', np.zeros_like(la)), ('rough', la)]:
        result[name+'_mean'] = float(np.exp(special.logsumexp(lw+li+slope)))
        result[name+'_log_energy'] = float(special.logsumexp(lw+2*li-lp+2*slope))
        result[name+'_sampled_max_log_gradient'] = float(np.max(li-lp+slope))
    result['quadrature_nodes'] = len(x)
    return result


def component_moments(mix, warp, n):
    mass, mean, second = 0., 0., 0.
    for m, s, prob in zip(mix.means, mix.sigmas, mix.weights):
        cuts = [-14,-10,-8,-6,-4,-2,-1,0,1,2,4,6,8,10,14]
        cuts += [(b-m)/s for seg in warp.segments for b in seg[:2] if -14 < (b-m)/s < 14]
        u, w = nodes(cuts, n); wg = prob*w*stats.norm.pdf(u); y = warp(m+s*u)
        mass += wg.sum(); mean += np.dot(wg, y); second += np.dot(wg, y*y)
    return float(mass), float(mean), float(second-mean*mean)






def theorem_constant(mix, warp, eta):
    s, bigs, weight = mix.sigmas.min(), mix.sigmas.max(), mix.weights.min()
    assert eta <= warp.r and warp.h < s
    return float(bigs/weight*np.exp((warp.r+eta)**2/(2*(s*s-warp.h*warp.h))))


def cases():
    out = []
    def add(group, delta, k=5, eta=.5, heterogeneous=False):
        mu = delta*(np.arange(k)-(k-1)/2)
        sig = np.array([.8,1.1,.9,1.2,1.]) if heterogeneous else np.ones(k)
        weights = np.array([.10,.15,.30,.25,.20]) if heterogeneous else np.ones(k)/k
        pattern = np.resize(PATTERN, k)
        centres = mu+eta*pattern
        out.append(({'group':group,'delta':float(delta),'components':k,'eta':float(eta),
                     'heterogeneous':heterogeneous,'means':mu.tolist(),'sigmas':sig.tolist(),
                     'weights':weights.tolist(),'rough_centres':centres.tolist(),
                     'radius':RADIUS,'bandwidth':BANDWIDTH},Mixture(mu,sig,weights),GapWarp(centres)))
    for delta in range(3,17): add('primary',delta)
    return out




def run():
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    rows=[]
    for idx,(cfg,mix,warp) in enumerate(cases()):
        fine = target_metrics(mix, warp, 256)
        cm1 = component_moments(mix, warp, 160)
        const=theorem_constant(mix,warp,cfg['eta'])
        assert fine['rough_sampled_max_log_gradient']<=np.log(const)+1e-10
        assert fine['rough_log_energy']<=2*np.log(const)+1e-10
        assert fine['rough_log_energy']>=np.log(cm1[2])-1e-8
        assert fine['rough_log_energy']<=fine['original_log_energy']+1e-10
        row={**{k:v for k,v in cfg.items() if not isinstance(v,list)},**fine,
             'original_variance':mix.variance(),'rough_variance':cm1[2],
             'original_log_normalized_energy':fine['original_log_energy']-np.log(mix.variance()),
             'rough_log_normalized_energy':fine['rough_log_energy']-np.log(cm1[2]),
             'log_energy_reduction':fine['original_log_energy']-fine['rough_log_energy'],
             'theorem_C':const,'minimum_log_slope':warp.minimum_log_slope(),
             'rough_mean':fine['rough_mean'],'warped_centre_span':float(np.ptp(warp(warp.centres))),
             'variance_matched_affine_log_energy':fine['original_log_energy']+np.log(cm1[2]/mix.variance())}
        rows.append(row)
        print(f"{idx+1:02}/{len(cases())}: {cfg['group']}, K={cfg['components']}, D={cfg['delta']:g}, eta={cfg['eta']:g}; "
              f"mean {fine['original_mean']:.5g}->{fine['rough_mean']:.5g}, log10 energy {fine['original_log_energy']/np.log(10):.4g}->{fine['rough_log_energy']/np.log(10):.4g}",flush=True)
    with (ROOT/'results/component_means_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def plot(data):
    apply_style()
    (ROOT/'figures').mkdir(parents=True,exist_ok=True)
    rows = sorted(
        (row for row in data
         if row['group'] == 'primary'),
        key=lambda row: row['delta'],
    )
    x = [row['delta'] for row in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.82)
    for name, color, marker, label in [
        ('original', RED, 'o', r'Original map $T$'),
        ('rough', BLUE, 's', r'Transformed map $S$'),
    ]:
        y = [math.exp(row[name + '_log_energy']) for row in rows]
        ax.plot(x, y, marker=marker, ms=2.3, lw=1.2, color=color, label=label)
    ax.set(
        yscale='log', xlabel=r'Separation between adjacent means $D$',
        ylabel=JACOBIAN_LABEL,
        xticks=[3, 6, 9, 12, 16], xlim=(3, 16),
    )
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc='upper center',
               frameon=False)
    fig.savefig(ROOT / 'figures/component_means_gradient_growth.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Plotted {len(rows)} freshly computed five-component cases.', flush=True)


def main():
    plot(run())


if __name__ == '__main__':
    main()
