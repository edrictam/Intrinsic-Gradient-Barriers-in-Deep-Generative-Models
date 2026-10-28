"""Generate Figure 1 from four normalizing-flow fits and fresh samples."""
from pathlib import Path
import sys
import os
import math
import time
import json
import csv
import tempfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'annals-density-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'annals-density-cache'))
import numpy as np
from scipy import special, optimize, stats, ndimage
import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)
SEED = 20261021
TARGETS = ['student', 'bimodal']
LOG2PI = math.log(2 * math.pi)
HROOT = math.sqrt(math.pi / 2)
SQRT2 = math.sqrt(2)
CONFIG = dict(seed=SEED, layers=4, hidden=[32, 32], max_log_scale=2.,
              batch_size=512, steps=800, learning_rate=1e-3, gradient_clip=5.,
              objective='reverse_KL', source='standard Gaussian', dtype='float64',
              threads=1, sample_count=100000, plot_bins=180, smoothing_bins=1.0,
              display_density='unconditional histogram / (N * bin area), smoothed',
              target_samples_used=0)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class GapWarp:
    """The paper's positive-slope 1D map, with analytic inverse in float64."""
    def __init__(self, centres, radius, bandwidth):
        self.centres = np.sort(np.asarray(centres, dtype=float))
        self.r, self.h = float(radius), float(bandwidth)
        cores = []
        for c in self.centres:
            lo, hi = c - self.r, c + self.r
            if cores and lo <= cores[-1][1]:
                cores[-1][1] = max(hi, cores[-1][1])
            else:
                cores.append([lo, hi])
        self.left, self.right = cores[0][0], cores[-1][1]
        self.segments = []
        prefix = self.left
        for i, (lo, hi) in enumerate(cores):
            self.segments.append((lo, hi, 'core', lo, prefix, prefix + hi - lo))
            prefix += hi - lo
            if i + 1 < len(cores):
                end = cores[i + 1][0]
                mid = (hi + end) / 2
                inc = self.h * HROOT * special.erf((mid - hi) / (SQRT2 * self.h))
                self.segments.append((hi, mid, 'left', hi, prefix, prefix + inc))
                prefix += inc
                self.segments.append((mid, end, 'right', end, prefix, prefix + inc))
                prefix += inc
        self.output_right = prefix
        self.shift = float(self._raw_numpy(np.array([0.]))[0])
        # A very deep gap cannot be evaluated by subtracting two integrals
        # accumulated from the left: both round to the same float. Anchor its
        # two halves locally and use erfc / Gaussian quantiles in the tails.
        self.central_gap = None
        for index, (lo, hi, kind, edge, _, _) in enumerate(self.segments):
            if kind != 'left':
                continue
            right = self.segments[index+1][1]
            if lo < 0 < right and -.5*((hi-lo)/self.h)**2 < math.log(1e-12):
                qm = self.h*HROOT*special.erfc((hi-lo)/(SQRT2*self.h))
                if 0 <= hi:
                    q0 = self.h*HROOT*special.erfc(-lo/(SQRT2*self.h))
                    ym = q0-qm
                else:
                    q0 = self.h*HROOT*special.erfc(right/(SQRT2*self.h))
                    ym = qm-q0
                self.central_gap = dict(left=lo, mid=hi, right=right,
                                        qmid=qm, ymid=ym, cleft=ym+qm, cright=ym-qm)
                break

    def _raw_numpy(self, x):
        x = np.asarray(x)
        y = np.empty_like(x)
        y[x <= self.left] = x[x <= self.left]
        y[x >= self.right] = self.output_right + x[x >= self.right] - self.right
        for lo, hi, kind, edge, prefix, _ in self.segments:
            m = (x >= lo) & (x <= hi)
            u = x[m]
            if kind == 'core':
                v = u - lo
            elif kind == 'left':
                v = self.h * HROOT * special.erf((u - edge) / (SQRT2 * self.h))
            else:
                v = self.h * HROOT * (special.erfc((edge-u)/(SQRT2*self.h))
                                      - special.erfc((edge-lo)/(SQRT2*self.h)))
            y[m] = prefix + v
        return y

    def numpy(self, x):
        x = np.asarray(x, dtype=float)
        y = self._raw_numpy(x)-self.shift
        gap = self.central_gap
        if gap is not None:
            m = (x >= gap['left']) & (x <= gap['mid'])
            y[m] = gap['cleft']-self.h*HROOT*special.erfc((x[m]-gap['left'])/(SQRT2*self.h))
            m = (x > gap['mid']) & (x <= gap['right'])
            y[m] = gap['cright']+self.h*HROOT*special.erfc((gap['right']-x[m])/(SQRT2*self.h))
        return y

    def forward(self, x):
        y = torch.zeros_like(x)
        mask = x <= self.left
        y[mask] = x[mask]
        mask = x >= self.right
        y[mask] = self.output_right + x[mask] - self.right
        for lo, hi, kind, edge, prefix, _ in self.segments:
            if self.central_gap is not None and kind != 'core' and self.central_gap['left'] <= lo < self.central_gap['right']:
                continue
            # Include endpoints without leaving uninitialized elements.
            mask = (x >= lo) & (x <= hi)
            u = x[mask]
            if kind == 'core':
                v = u-lo
            elif kind == 'left':
                v = self.h*HROOT*torch.erf((u-edge)/(SQRT2*self.h))
            else:
                v = self.h*HROOT*(torch.erfc((edge-u)/(SQRT2*self.h))
                                 - special.erfc((edge-lo)/(SQRT2*self.h)))
            y[mask] = prefix + v
        out = y-self.shift
        gap = self.central_gap
        if gap is not None:
            m = (x >= gap['left']) & (x <= gap['mid'])
            out[m] = gap['cleft']-self.h*HROOT*torch.erfc((x[m]-gap['left'])/(SQRT2*self.h))
            m = (x > gap['mid']) & (x <= gap['right'])
            out[m] = gap['cright']+self.h*HROOT*torch.erfc((gap['right']-x[m])/(SQRT2*self.h))
        return out

    def inverse(self, y):
        v = y + self.shift
        x = torch.zeros_like(y)
        mask = v <= self.left
        x[mask] = v[mask]
        mask = v >= self.output_right
        x[mask] = self.right + v[mask] - self.output_right
        for lo, hi, kind, edge, prefix, output_end in self.segments:
            if self.central_gap is not None and kind != 'core' and self.central_gap['left'] <= lo < self.central_gap['right']:
                continue
            mask = (v >= prefix) & (v <= output_end)
            u = v[mask]-prefix
            if kind == 'core':
                part = lo+u
            elif kind == 'left':
                part = edge+SQRT2*self.h*torch.erfinv(u/(self.h*HROOT))
            else:
                arg = special.erf((edge-lo)/(SQRT2*self.h))-u/(self.h*HROOT)
                part = edge-SQRT2*self.h*torch.erfinv(arg)
            x[mask] = part
        gap = self.central_gap
        if gap is not None:
            m = (y >= gap['cleft']-self.h*HROOT) & (y <= gap['ymid'])
            p = (gap['cleft']-y[m])/(2*self.h*HROOT)
            x[m] = gap['left']-self.h*torch.special.ndtri(p)
            m = (y > gap['ymid']) & (y <= gap['cright']+self.h*HROOT)
            p = (y[m]-gap['cright'])/(2*self.h*HROOT)
            x[m] = gap['right']+self.h*torch.special.ndtri(p)
        return x

    def log_slope(self, x):
        c = torch.as_tensor(self.centres, dtype=x.dtype)
        d = torch.abs(x[:, None] - c).min(dim=1).values
        inside = -.5*(torch.clamp(d-self.r, min=0.)/self.h)**2
        return torch.where((x < self.left) | (x > self.right), 0., inside)

    def parameters(self):
        return dict(centres=self.centres.tolist(), radius=self.r, bandwidth=self.h,
                    minimum_slope=float(min([1.] + [math.exp(-.5*((hi-lo)/self.h)**2)
                                        for lo, hi, k, *_ in self.segments if k == 'left'])))


class Target:
    def __init__(self, name):
        self.name = name
        self.df = 2.1
        self.means = None
        if name == 'student':
            self.title = 'Student t (2.1 degrees of freedom)'
            self.limit = 12.5
            self.tail_scale = math.sqrt(self.df)
        else:
            if name == 'bimodal':
                self.title = 'Bimodal (centres at -10 and 10)'
                self.means = np.array([[-10., 0.], [10., 0.]])
                self.sigma = .65
                self.limit = 12.5
            else:
                raise ValueError(name)
            pattern = np.array([[.8, -.6], [-.7, .9]])
            self.rough_means = self.means+.25*self.sigma*pattern[:len(self.means)]
            self.warps = [GapWarp(self.rough_means[:, j], 1.5*self.sigma,
                                  .9*self.sigma) for j in range(2)]
        self.quantiles = np.array([[self.marginal_quantile(q,j) for j in range(2)]
                                  for q in [.25, .5, .75]])
        self.affine = {}
        for after in [False, True]:
            qs = self.transform_numpy(self.quantiles) if after else self.quantiles
            self.affine[after] = (qs[1], (qs[2]-qs[0])/1.3489795003921634)

    def marginal_quantile(self, q, j):
        if self.means is None:
            return float(stats.t.ppf(q, self.df))
        def cdf(x):
            return special.ndtr((x-self.means[:, j])/self.sigma).mean()-q
        return optimize.brentq(cdf, self.means[:,j].min()-12*self.sigma,
                               self.means[:,j].max()+12*self.sigma, xtol=1e-13)

    def log_prob(self, x):
        if self.means is None:
            constant = special.gammaln((self.df+1)/2)-special.gammaln(self.df/2)\
                       -.5*math.log(self.df*math.pi)
            return (constant-(self.df+1)/2*torch.log1p(x*x/self.df)).sum(dim=1)
        means = torch.as_tensor(self.means, dtype=x.dtype)
        d2 = ((x[:,None,:]-means)/self.sigma).square().sum(dim=2)
        return torch.logsumexp(-.5*d2, dim=1)-math.log(len(self.means))\
               -LOG2PI-2*math.log(self.sigma)

    def transform_numpy(self, x):
        if self.means is None:
            return np.arcsinh(x/self.tail_scale)
        return np.stack([w.numpy(x[:,j]) for j,w in enumerate(self.warps)], axis=1)

    def transform(self, x):
        if self.means is None:
            return torch.asinh(x/self.tail_scale)
        return torch.stack([w.forward(x[:,j]) for j,w in enumerate(self.warps)], dim=1)

    def inverse_transform(self, y):
        if self.means is None:
            x = self.tail_scale*torch.sinh(y)
            ay = y.abs()
            logdet = (math.log(self.tail_scale)+ay+torch.nn.functional.softplus(-2*ay)
                      - math.log(2)).sum(dim=1)
            return x, logdet
        x = torch.stack([w.inverse(y[:,j]) for j,w in enumerate(self.warps)], dim=1)
        logdet = -self.log_transform_det(x)
        return x, logdet

    def log_transform_det(self, x):
        if self.means is None:
            return (-.5*torch.log(self.tail_scale**2+x*x)).sum(dim=1)
        return torch.stack([w.log_slope(x[:,j]) for j,w in enumerate(self.warps)], dim=1).sum(dim=1)

    def specification(self):
        result = dict(name=self.name, title=self.title, display_limit=self.limit,
                      affine={str(k):dict(median=m.tolist(), scale=s.tolist())
                              for k,(m,s) in self.affine.items()})
        if self.means is None:
            result.update(df=self.df, independent_coordinates=True, asinh_scale=self.tail_scale)
        else:
            result.update(means=self.means.tolist(), sigma=self.sigma,
                          weights=[1/len(self.means)]*len(self.means),
                          approximate_means=self.rough_means.tolist(),
                          transformation=[w.parameters() for w in self.warps])
        return result


class Coupling(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.i = layer % 2
        self.j = 1-self.i
        self.net = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,32),
                                 nn.Tanh(), nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def st(self, fixed):
        s,t = self.net(fixed[:,None]).unbind(dim=1)
        return 2*torch.tanh(s/2),t

    def forward(self, z):
        s,t = self.st(z[:,self.i])
        out = torch.stack([z[:,self.i], z[:,self.j]*s.exp()+t], dim=1)
        if self.i == 1:
            out = out.flip(dims=[1])
        return out,s

    def inverse(self, x):
        s,t = self.st(x[:,self.i])
        out = torch.stack([x[:,self.i], (x[:,self.j]-t)*(-s).exp()], dim=1)
        if self.i == 1:
            out = out.flip(dims=[1])
        return out,-s


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Coupling(i) for i in range(4)])

    def forward(self, z):
        ld = z.new_zeros(len(z))
        for layer in self.layers:
            z,delta = layer(z)
            ld = ld+delta
        return z,ld

    def inverse(self, x):
        ld = x.new_zeros(len(x))
        for layer in reversed(self.layers):
            x,delta = layer.inverse(x)
            ld = ld+delta
        return x,ld


def generate(flow, target, after, z):
    u,ld = flow(z)
    med,scale = [torch.as_tensor(a) for a in target.affine[after]]
    y = med+scale*u
    ld = ld+scale.log().sum()
    if after:
        x,delta = target.inverse_transform(y)
        ld = ld+delta
    else:
        x = y
    logq = -.5*z.square().sum(dim=1)-LOG2PI-ld
    return x,logq


def model_log_prob(flow, target, after, x):
    y = target.transform(x) if after else x
    med,scale = [torch.as_tensor(a) for a in target.affine[after]]
    z,ild = flow.inverse((y-med)/scale)
    logq = -.5*z.square().sum(dim=1)-LOG2PI+ild-scale.log().sum()
    if after:
        logq = logq+target.log_transform_det(x)
    return logq


def fit(target, after, steps):
    torch.manual_seed(SEED)
    flow = Flow()
    opt = torch.optim.Adam(flow.parameters(), lr=CONFIG['learning_rate'])
    gen = torch.Generator().manual_seed(SEED+1)
    start = time.perf_counter()
    for step in range(steps):
        z = torch.randn(CONFIG['batch_size'], 2, generator=gen)
        x,logq = generate(flow,target,after,z)
        loss = (logq-target.log_prob(x)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss: {target.name}, {after}, {step}')
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), CONFIG['gradient_clip'],
                                          error_if_nonfinite=True)
        opt.step()
    elapsed = time.perf_counter()-start
    return flow,elapsed










def train():
    """Train the four models used in Figure 1."""
    fits = {}
    for name in TARGETS:
        for after in [False, True]:
            model, elapsed = fit(Target(name), after, CONFIG['steps'])
            model.eval()
            fits[name, after] = model, dict(
                training_seconds=elapsed,
                parameter_count=sum(p.numel() for p in model.parameters()))
            branch = 'after' if after else 'before'
            print(f"Trained {name}, {branch}: {CONFIG['steps']} updates", flush=True)
    return fits


@torch.no_grad()
def batched_density(model,target,after,points):
    return np.concatenate([model_log_prob(model,target,after,torch.as_tensor(p)).exp().numpy()
                           for p in np.array_split(points,max(1,math.ceil(len(points)/8192)))])


def exact_diagnostic_probabilities(target):
    if target.means is None:
        threshold=float(stats.t.ppf(.995,target.df))
        return dict(tail_threshold=threshold, true_tail_probability=.01)
    # Union of disjoint radius-2*sigma disks around true component centres.
    rad=2*target.sigma
    means=target.means
    k=len(means)
    distances=((means[:,None,:]-means[None,:,:])**2).sum(axis=2)
    assert np.min(distances+np.eye(k)*1e10)>(2*rad)**2
    nc=distances/target.sigma**2
    per_disk=stats.ncx2.cdf(4,2,nc).sum(axis=1)/k
    return dict(neighbourhood_radius=rad, true_per_mode_disk_probabilities=per_disk.tolist(),
                true_outside_mode_disks=float(1-per_disk.sum()))


def present(fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,
                         'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#bcc7cc','axes.linewidth':.7,
                         'axes.labelcolor':'#354b57','text.color':'#243e4c',
                         'xtick.color':'#4d606a','ytick.color':'#4d606a',
                         'figure.facecolor':'white','axes.facecolor':'white',
                         'pdf.fonttype':42,'ps.fonttype':42})
    palette=['#ffffff','#f3f8ed','#dfefce','#b9e1cf','#83c9c4','#4ca6b6','#2b799f','#225d8d']
    cmap=LinearSegmentedColormap.from_list('light_blue_green',palette,N=256)
    cmap.set_under('white')
    target_color='#b56c25'
    out=FIGURES
    out.mkdir(parents=True,exist_ok=True)
    # Display limits are presentation settings, independent of trained weights.
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(10.2,9.0),layout='constrained')
    fig.get_layout_engine().set(w_pad=.06,h_pad=.10,hspace=.04,wspace=.02)
    norm=LogNorm(vmin=1e-5,vmax=.22)
    for row,name in enumerate(TARGETS):
        target=Target(name)
        lim=target.limit
        bins=CONFIG['plot_bins']
        edges=np.linspace(-lim,lim,bins+1)
        centres=(edges[:-1]+edges[1:])/2
        cell=(edges[1]-edges[0])**2
        xx,yy=np.meshgrid(centres,centres)
        points=np.stack([xx.ravel(),yy.ravel()],axis=1)
        with torch.no_grad():
            true=target.log_prob(torch.as_tensor(points)).exp().numpy().reshape(xx.shape)
        diag=exact_diagnostic_probabilities(target)
        levels=np.array([.0003,.003,.03]) if name=='student' else np.array([.003,.03,.10])
        for col,after in enumerate([False,True]):
            model,payload=fits[name,after]
            gen=torch.Generator().manual_seed(SEED+100)
            # Only source/flow draws: no target sampler is used for any diagnostic.
            source=torch.randn(CONFIG['sample_count'],2,generator=gen)
            with torch.no_grad():
                chunks=[generate(model,target,after,z) for z in source.split(8192)]
                samples=torch.cat([c[0] for c in chunks]).numpy()
                logq=torch.cat([c[1] for c in chunks])
                logp=target.log_prob(torch.as_tensor(samples))
                kl=(logq-logp).numpy()
            assert np.isfinite(samples).all() and np.isfinite(kl).all()
            branch='after' if after else 'before'
            hist,_,_=np.histogram2d(samples[:,0],samples[:,1],bins=(edges,edges))
            density=ndimage.gaussian_filter(hist.T/(len(samples)*cell),CONFIG['smoothing_bins'],
                                           mode='constant',cval=0.)
            qgrid=batched_density(model,target,after,points).reshape(xx.shape)
            result=dict(target=name,branch=branch,training_seconds=payload['training_seconds'],
                        steps=CONFIG['steps'],parameter_count=payload['parameter_count'],
                        reverse_kl_mc=float(kl.mean()),reverse_kl_mc_se=float(kl.std(ddof=1)/math.sqrt(len(kl))),
                        fraction_outside_display=float(np.any(np.abs(samples)>lim,axis=1).mean()),
                        model_box_mass_grid=float(qgrid.sum()*cell),
                        target_box_mass_grid=float(true.sum()*cell),
                        half_L1_inside_display=float(.5*np.abs(qgrid-true).sum()*cell))
            if name=='student':
                result.update(diag)
                result['estimated_tail_probability']=float((np.abs(samples[:,0])>diag['tail_threshold']).mean())
            else:
                d2=((samples[:,None,:]-target.means[None,:,:])**2).sum(axis=2)
                masses=(d2<diag['neighbourhood_radius']**2).mean(axis=0)
                result.update(diag)
                result.update(estimated_per_mode_disk_probabilities=masses.tolist(),
                              estimated_outside_mode_disks=float(1-masses.sum()),
                              nearest_centre_fractions=np.bincount(d2.argmin(axis=1),minlength=len(target.means)).astype(float).tolist())
                result['nearest_centre_fractions']=[v/len(samples) for v in result['nearest_centre_fractions']]
            rows.append(result)
            ax=axes[row,col]
            im=ax.pcolormesh(edges,edges,np.maximum(density,1e-20),cmap=cmap,norm=norm,
                             shading='flat',rasterized=True)
            ax.contour(xx,yy,true,levels=levels,colors=target_color,linewidths=1.0,
                       linestyles='dashed',alpha=.95)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel=r'$x_1$',ylabel=r'$x_2$')
            heading='Before transformation' if not after else 'After transformation'
            ax.set_title(f'({chr(97+row*2+col)})  {heading}',loc='left',fontsize=17,fontweight='normal',pad=14)
            ax.tick_params(labelsize=16,length=4.5,width=.8)
            if col==0:
                label='Student t, DF = 2.1' if name=='student' else 'Bimodal'
                ax.text(-.39,.5,label,transform=ax.transAxes,rotation=90,ha='center',va='center',
                        fontsize=18,fontweight='normal')
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.56,pad=.025,aspect=28,
                    anchor=(0.,.5))
    cb.set_label('Estimated density (log scale)',fontsize=16,labelpad=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=16,width=.7,length=3)
    fig.legend(handles=[Patch(facecolor='#83c9c4',edgecolor='none',label='Generated density'),
                        Line2D([0],[0],color=target_color,lw=1.2,ls='--',label='Target contours')],
               loc='upper center',bbox_to_anchor=(.51,1.065),ncol=2,frameon=False,
               fontsize=16,handlelength=2.2,columnspacing=2.0)
    fig.savefig(out/'density_comparison_2x2.pdf',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    dump(RESULTS/'density_metrics.json',rows)
    with (RESULTS/'density_metrics.csv').open('w',newline='') as file:
        keys=['target','branch','training_seconds','steps','reverse_kl_mc','reverse_kl_mc_se',
              'half_L1_inside_display','fraction_outside_display']
        writer=csv.DictWriter(file,keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


def main():
    present(train())


if __name__ == '__main__':
    main()
