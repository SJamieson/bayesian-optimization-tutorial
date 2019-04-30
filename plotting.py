from typing import Optional, List, Tuple, Dict
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.contour import QuadContourSet, ClabelText
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.mlab as mlab
import numpy as np
import warnings


def caldera_sim_function(x, y):
    warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
    x, y = x / 10.0, y / 10.0
    z0 = mlab.bivariate_normal(x, y, 10.0, 5.0, 5.0, 0.0)
    z1 = mlab.bivariate_normal(x, y, 1.0, 2.0, 2.0, 5.0)
    z2 = mlab.bivariate_normal(x, y, 1.7, 1.7, 8.0, 8.0)
    return 50000.0 * z0 + 2500.0 * z1 + 5000.0 * z2


def draw_caldera_maxima(axes: plt.Axes):
    maxima = list()
    maxima.extend(axes.plot(20, 46, 'bx'))
    maxima.extend(axes.plot(79, 79, 'bx'))
    return maxima


class RobotPlot:
    def __init__(self, ax, title):
        self.ax: plt.Axes = ax
        self.pos: Dict[int, Optional[Tuple[int, int]]] = dict()
        self.robot_marker: Dict[int, List[plt.Line2D]] = dict()
        self.path: Dict[int, List[List[plt.Line2D]]] = dict()
        self.ax.title.set_text(title)

    def draw_robot(self, new_pos, connect=True, index=0):
        for marker in self.robot_marker.get(index, list()):
            marker.remove()
        changed = list()
        if connect and self.pos.get(index, None) is not None:
            if index not in self.path:
                self.path[index] = list()
            self.path[index].append(self.ax.plot([self.pos[index][0], new_pos[0]],
                                                 [self.pos[index][1], new_pos[1]], color='k'))
            changed.extend(self.path[index][-1])
        self.robot_marker[index] = self.ax.plot(*new_pos, '*m')
        self.pos[index] = new_pos
        changed.extend(self.robot_marker[index])
        return changed


class ContourPlot(RobotPlot):
    def __init__(self, ax, title):
        super().__init__(ax, title)
        self.contours: Optional[QuadContourSet] = None
        self.contour_labels: List[ClabelText] = list()
        self.cbar: Optional[Colorbar] = None

    def draw_contours(self, X, Y, Z, label=True, colorbar=False, **contour_kw):
        if self.contours is not None:
            for coll in self.contours.collections:
                coll.remove()
            for label in self.contour_labels:
                label.remove()
            if self.cbar is not None:
                self.cbar.remove()
                self.cbar = None
        self.contours = self.ax.contour(X, Y, Z, **contour_kw)
        changed = [self.contours]
        if label:
            self.contour_labels = self.ax.clabel(self.contours, inline=1, fontsize=8, fmt='%.3g')
            changed.append(self.contour_labels)
        if colorbar and len(self.contour_labels) > 0:
            self.cbar = plt.gcf().colorbar(self.contours, ax=self.ax, fraction=0.046, pad=0.04)
            changed.append(self.cbar)
        return changed


class HeatmapPlot(RobotPlot):
    def __init__(self, ax, title):
        super().__init__(ax, title)
        self.im: Optional[AxesImage] = None
        self.cbar: Optional[Colorbar] = None

    def draw_heatmap(self, map, colorbar=True, **heatmap_kw):
        if self.cbar is not None:
            self.cbar.remove()
        self.im = self.ax.imshow(map, interpolation='nearest', **heatmap_kw)
        self.ax.invert_yaxis()
        changed = [self.im]
        if colorbar:
            self.cbar = plt.gcf().colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            # changed.append(self.cbar)
        return changed


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
