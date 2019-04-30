import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


class FunctionPlot():
    def __init__(self, ax, title, domain, range=None, resolution=None):
        self.locked = []
        self.patches = []
        self.ax: plt.Axes = ax
        self.ax.title.set_text(title)
        self.domain = domain
        self.range = range
        self.num_points = ((domain[1] - domain[0]) / resolution) if resolution is not None else 100
        self.x = np.atleast_2d(np.linspace(domain[0], domain[1], self.num_points)).T
        if domain is not None and range is not None:
            self.ax.axis([domain[0], domain[1], range[0], range[1]])

    def add_function(self, name, f, style=None, f_sigma_style=None, locked=False, **kwargs):
        new_patches = []
        y = f(self.x)
        if isinstance(y, tuple):
            y, sigma = y
        else:
            y, sigma = y, None
        y = y.reshape((-1,))
        if style is None:
            new_patches.extend(self.ax.plot(self.x, y, label=name, **kwargs))
        else:
            new_patches.extend(self.ax.plot(self.x, y, style, label=name, **kwargs))
        if sigma is not None:
            sigma = sigma.reshape((-1,))
            xfill = np.concatenate([self.x, self.x[::-1]])
            yfill = np.concatenate([y - 1 * sigma, (y + 1 * sigma)[::-1]])
            new_patches.extend(self.ax.fill(xfill, yfill, alpha=.5, fc=f_sigma_style, ec='None',
                                            label=f'{name} $\sigma$-bound' if name is not None else None))
        self.ax.legend()
        if locked:
            self.locked.append(new_patches)
            return None
        self.patches.append(new_patches)
        return len(self.patches) - 1

    def remove(self, id):
        for l in self.patches[id]:
            l.remove()
        self.patches.pop(id)

    def reset(self):
        for i in range(len(self.patches) - 1, -1, -1):
            self.remove(i)

    def mark_point(self, x, y, style, markersize=5, **kwargs):
        patches = self.ax.plot(x, y, style, markersize=markersize, **kwargs)
        self.patches.append(patches)
        return len(self.patches) - 1

    def vline(self, x, style, **kwargs):
        patches = self.ax.plot((x, x), self.range, style, **kwargs)
        self.patches.append(patches)
        return len(self.patches) - 1

    def hline(self, y, style, **kwargs):
        patches = self.ax.plot(self.domain, (y, y), style, **kwargs)
        self.patches.append(patches)
        return len(self.patches) - 1

    def add_normal_pdf(self, x, y, sigma, style, highlight_above=None, highlight_below=None, highlight_color=None,
                       vline=True, **kwargs):
        patches = []
        X = np.atleast_2d(np.linspace(self.range[0], self.range[1], self.num_points)).T
        Y = stats.norm.pdf(X, y, sigma)
        patches.extend(self.ax.plot(Y + x, X, style, **kwargs))
        if vline:
            patches.extend(self.ax.plot((x, x), self.range, style, alpha=0.5))
        if highlight_above is not None or highlight_below is not None:
            if highlight_above is not None:
                Xfill = X[X >= highlight_above]
                Yfill = Y[X >= highlight_above]
            else:
                Xfill = X[X <= highlight_below]
                Yfill = Y[X <= highlight_below]
            xfill = np.concatenate([Yfill + x - Yfill, (Yfill + x)[::-1]])
            yfill = np.concatenate([Xfill, Xfill[::-1]])
            patches.extend(self.ax.fill(xfill, yfill, fc=highlight_color, alpha=0.5, ec='None'))

        self.patches.append(patches)
        return len(self.patches) - 1
