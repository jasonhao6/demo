"""Hierarchical Risk Parity Algorithm
1. Hierarchical Risk Parity (using tree structure)
2. Hierarchy Risk Parity (using bi-section, original implementation by Marcos Lopez de Prado)
3. Simple Risk Parity (Inverse Volatility)
4. Min Variance Optimization

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
"""

import collections
import datetime
import dateutil
import logging
import os

import cvxopt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as scich

from matplotlib import pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


# ### ### ### Help Functions ### ### ### #

def get_sub_cluster_var(cov, symbols):
    """Compute the sub-cluster variance as w * Var * w, where w is inverse variance of each symbol."""
    sub_cov = cov.loc[symbols, symbols]
    inv_var = np.reciprocal(np.diag(sub_cov.values))
    inv_var_w = np.divide(inv_var, inv_var.sum())
    out_var = np.matmul(inv_var_w, np.matmul(sub_cov.values, inv_var_w))
    return out_var


def bisector_var_weight(cov, symbols):
    """Recursive bisector weight allocation based on input covariance matrix. """
    weight = pd.Series(1, index=symbols)
    c_items = [symbols]  # list of items, each item itself is a lists of symbols/tickers

    while len(c_items) > 0:
        # bi-section, item [list of symbols] drops out if len(item) = 1
        c_items = [item[j:k] for item in c_items for j, k in ((0, len(item) // 2), (len(item) // 2, len(item))) if
                   len(item) > 1]
        for i in range(0, len(c_items), 2):  # parse in pairs
            left = c_items[i]  # left cluster
            right = c_items[i + 1]  # right cluster
            left_var = get_sub_cluster_var(cov, left)
            right_var = get_sub_cluster_var(cov, right)
            alpha = right_var / (left_var + right_var)
            weight[left] *= alpha
            weight[right] *= 1 - alpha
    return weight


def corr_dist_prabo(v, v2):
    """Compute the correlation distance sqrt[0.5*(1-corr)] between two return vectors, defined by Lopez de Prado . """
    r = np.corrcoef(v, v2)
    r = r[0, 1]  # 2x2 array to off-diagonal element
    return np.sqrt(0.5 * (1 - r))


# ### ### ### Hierarchy Risk Parity ### ### ### #

class HierarchyRiskParity:
    """Hierarchy Risk Parity.

    Input time-series Data Frame with columns as tickers.
    """

    PARENT = 'parent'
    LEFT = 'left'
    RIGHT = 'right'
    NC = 'n'  # Number of points in a cluster
    DIST = 'dist'

    DEFAULT_METHOD = 'average'

    def __init__(self, data, method=None, metric=None):
        # input data
        self._data = data  # return data

        # key parameters (of Hierarchical cluster)
        self.method = self.DEFAULT_METHOD if method is None else self.DEFAULT_METHOD
        self.metric = corr_dist_prabo if metric is None else metric

        # Intermediate steps
        self._z = None  # naive SciPi linkage output in Numpy array
        self._nodes2d = None  # list of nodes, last one is the final nodes
        self._weight2d = None  # weight in each iteration of hierarchical tree structure, 1st one is final weight

        # Key output with corresponding properties
        self._linkage = None  # formatted linkage output in DataFrame
        self._nodes = None
        self._weight = None
        return

    @property
    def data(self):
        return self._data

    @property
    def symbols(self):
        return self._data.columns.tolist()

    @property
    def linkage(self):
        if self._linkage is None:
            self.build_cluster_tree()
        return self._linkage

    @property
    def nodes(self):
        """Symbols ordered based on hierarchical cluster tree structure. """
        if self._nodes is None:
            self.allocate_weight()
        return self._nodes

    @property
    def weight(self):
        if self._weight is None:
            self.allocate_weight()
        return self._weight

    def run(self):
        # Hierarchical Tree Clustering, default method='average', metric=self.corr_dist
        self.build_cluster_tree()

        # Weight allocation based on the tree structure of Hierarchical clustering
        self.allocate_weight()
        return

    def build_cluster_tree(self):
        y = self.data.transpose().values
        z = scich.linkage(y, method=self.method, metric=self.metric)
        self._z = z

        df_link = pd.DataFrame(z[:, [0, 1]].astype(int), columns=[self.LEFT, self.RIGHT])
        df_link[self.NC] = pd.Series(z[:, 3].astype(int))
        df_link[self.PARENT] = [y.shape[0] + x for x in range(z.shape[0])]
        df_link[self.DIST] = pd.Series(z[:, 2].astype(float))  # float
        self._linkage = df_link
        return

    def allocate_weight(self):
        cov = self.data.cov()
        link = self.linkage[[self.LEFT, self.RIGHT, self.NC, self.PARENT]]

        self._weight2d, self._nodes2d = self.allocate_var_weight_tree(link, cov)

        int2tix = collections.OrderedDict([(i, x) for i, x in enumerate(self.symbols)])
        self._nodes = [int2tix[i] for i in self._nodes2d[-1]]
        self._weight = self._weight2d.iloc[0].copy()
        return

    @classmethod
    def allocate_var_weight_tree(cls, link, cov):
        # nodes = list of positional integers
        node2n = link.set_index(cls.PARENT)[cls.NC].to_dict()
        top_node = link[cls.PARENT].iloc[-1]
        nodes = [top_node] * node2n[top_node]

        # initialization
        symbols = cov.columns.tolist()
        weight = pd.Series(1, symbols)
        nodes2d = []
        weight2d = pd.DataFrame(columns=symbols, index=range(len(link.index)), dtype=float)

        for r, row in link.iloc[::-1].iterrows():  # r from large to 0
            left = row[cls.LEFT]
            right = row[cls.RIGHT]
            left_n = node2n.get(left, 1)
            right_n = node2n.get(right, 1)
            child_nodes = [left] * left_n + [right] * right_n

            # update nodes (list of position integers)
            idx = nodes.index(row[cls.PARENT])
            nodes[idx:idx + row[cls.NC]] = child_nodes

            # extract symbols from child positions
            left_syms = symbols[idx:idx + left_n]
            right_syms = symbols[idx + left_n:idx + left_n + right_n]

            # compute variance within each child cluster (recursively)
            # this can be replaced by volatility or something else
            left_var = get_sub_cluster_var(cov, left_syms)
            right_var = get_sub_cluster_var(cov, right_syms)

            # across-cluster allocation
            alpha = right_var / (left_var + right_var)
            weight[left_syms] *= alpha
            weight[right_syms] *= 1 - alpha

            # output updates
            nodes2d.append(nodes.copy())
            weight2d.iloc[r] = weight.copy()

        return weight2d, nodes2d

    def dendrogram(self, labels=None):
        """Plot dendrogram using scipy.cluster.hierachy.dendrogram, with labels."""
        labels = self.symbols if labels is None else labels
        return scich.dendrogram(self._z, labels=labels)

    def heatmap(self, x=None, figsize=(16, 6), title='', sym2name=None, **kwargs):
        """Plot heatmap of an N x N matrix x, ordered by Hiearchical cluster tree nodes.
        If x is None, it defaults to the correlation matrix of input data self.data.
        """
        if x is None:
            x = self._data.corr()
        nodes = self._nodes
        x = x.loc[nodes, nodes]

        plt.figure(figsize=figsize)
        ax = sns.heatmap(x, cmap='RdYlGn', **kwargs)
        if title:
            ax.set_title(title, fontsize=22)
        if isinstance(sym2name, dict):
            labels = [y.get_text() for y in ax.get_yticklabels()]
            labels = [sym2name.get(tix, tix) for tix in labels]
            ax.set_yticklabels(labels, rotation=0)
        return ax

    def __repr__(self):
        return 'Hierarchy Risk Parity (method={}, metric={})'.format(self.method, self.metric)


class HierarchyRiskParityPrado(HierarchyRiskParity):
    """Original Hierarchy Risk Parity implementation by Marcos Lopez de Prado.

    Lopez de Prado's original approach ignores the tree structure of the hierarchical clustering.
    Instead he uses bisector partition on bottome-level nodes to allocate weight.

    The algorithm is broken down into 3 steps:
    1. Hierarchical tree clustering
    2. Quasi-diagonalising (bottom level node re-order)
    3. Bisection weight allocation (no use of tree structure obtained in step 1)
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        return

    @property
    def nodes(self):
        """Symbols ordered based on hierarchical cluster tree structure. """
        if self._nodes is None:
            self.quasi_diagonalise()
        return self._nodes

    def run(self):
        # Step 1: Hierarchical Tree Clustering, default method='average', metric=self.corr_dist
        self.build_cluster_tree()

        # Step 2: Quasi-Diagonalising
        self.quasi_diagonalise()  # this is a redundant step, can be extracted from dendrogram

        # Step 3: Weight Allocation (Variance)
        self.allocate_weight()

        return

    def quasi_diagonalise(self):
        link = self.linkage[[self.LEFT, self.RIGHT, self.NC, self.PARENT]]  # all integer
        nodes2d = self.reorder_linkage_nodes(link)
        self._nodes2d = nodes2d

        int2tix = collections.OrderedDict([(i, x) for i, x in enumerate(self.symbols)])
        self._nodes = [int2tix[i] for i in nodes2d[-1]]
        return

    def allocate_weight(self):
        nodes = self.nodes
        cov = self.data.cov()
        self._weight = bisector_var_weight(cov, nodes)
        return

    @classmethod
    def reorder_linkage_nodes(cls, link):
        node2n = link.set_index(cls.PARENT)[cls.NC].to_dict()
        top_node = link[cls.PARENT].iloc[-1]

        nodes = [top_node] * node2n[top_node]
        nodes2d = []
        for _, row in link.iloc[::-1].iterrows():
            left = row[cls.LEFT]
            right = row[cls.RIGHT]
            child_nodes = [left] * node2n.get(left, 1) + [right] * node2n.get(right, 1)
            idx = nodes.index(row[cls.PARENT])
            nodes[idx:idx+row[cls.NC]] = child_nodes
            nodes2d.append(nodes.copy())
        return nodes2d

    @classmethod
    def reorder_linkage_nodes2(cls, link):
        """An alternative implementation of re-order linkage nodes from a tree structure.
        No intermediate step, only final nodes. """
        root_node = link[cls.PARENT].iloc[-1]
        nodes = [root_node]

        for _, row in link.iloc[::-1].iterrows():
            idx = nodes.index(row[cls.PARENT])
            nodes = nodes[:idx] + [row[cls.LEFT], row[cls.RIGHT]] + nodes[1 + idx:]

        # output list of nodes, to be consistent with main reorder_link_nodes()
        return [nodes]

    def __repr__(self):
        return 'Hierarchy Risk Parity with Bisection Weight Allocation (method={}, metric={})'.format(
            self.method, self.metric
        )


# ### ### ### Volatility Help Function ### ### ### #

def get_year_frac(start_date, end_date):
    """Get number of year in fraction between two dates.
    This implementation is an estimate, revisit here Jason.
    """
    delta = dateutil.relativedelta.relativedelta(end_date, start_date)
    return delta.years + delta.months / 12 + delta.days / 365.25


def compute_volatility_from_returns(data, start=None, end=None):
    start = data.first_valid_index() if start is None else start
    end = data.last_valid_index() if end is None else end
    num_years = get_year_frac(start, end)
    periods_per_year = len(data.index) / num_years
    return np.sqrt(periods_per_year) * data.std()


def compute_cagr(data, start=None, end=None):
    """Input close price data. """
    start = data.first_valid_index() if start is None else start
    end = data.last_valid_index() if end is None else end
    num_years = get_year_frac(start, end)

    first_index = data.first_valid_index()
    last_index = data.last_valid_index()
    ratio = np.divide(data.loc[last_index], data.loc[first_index])
    cagr = np.power(ratio, 1 / num_years) - 1
    return cagr


# ### ### ### Inverse Volatility Risk Parity ### ### ### #

class InvVolRiskParity:
    """Traditional Inverse Volatility Risk Parity

    Input time-series Data Frame with columns as tickers
    """

    def __init__(self, data, start=None, end=None):
        start = data.first_valid_index() if start is None else pd.to_datetime(start)
        end = data.last_valid_index() if end is None else pd.to_datetime(end)

        self._data = data[start:end]
        self.start = start
        self.end = end

        self._vol = None
        self._weight = None
        return

    @property
    def data(self):
        return self._data

    @property
    def symbols(self):
        return self._data.columns.tolist()

    @property
    def weight(self):
        return self._weight

    def run(self):
        vol = compute_volatility_from_returns(self.data)
        inv_vol = 1 / vol
        weight = inv_vol / inv_vol.sum()

        self._vol = vol
        self._weight = weight
        return


# ### ### ### Min Variance Portfolio ### ### ### #

def get_min_var_portfolio(cov, n_iters=100, show_progress=False):
    n = len(cov)
    mus = [10 ** (5.0 * t / n_iters - 1.0) for t in range(n_iters)]

    # cvxopt matrices
    s = cvxopt.matrix(cov)
    pbar = cvxopt.matrix(np.ones(cov.shape[0]))

    # Constraint matrices
    g = -cvxopt.matrix(np.eye(n))  # negative n x n identity matrix
    h = cvxopt.matrix(0.0, (n, 1))
    a = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    # calculate efficient frontier weights using quadratic programming
    portfolios = []
    cvxopt.solvers.options['show_progress'] = show_progress
    for mu in mus:
        port = cvxopt.solvers.qp(mu * s, -pbar, g, h, a, b)
        portfolios.append(port['x'])

    # Efficient frontier risks and returns
    returns = [cvxopt.blas.dot(pbar, x) for x in portfolios]
    vols = [np.sqrt(cvxopt.blas.dot(x, s * x)) for x in portfolios]

    # 2nd degree polynomial fit off frontier curve
    m1 = np.polyfit(returns, vols, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # optimal portfolios
    wt = cvxopt.solvers.qp(cvxopt.matrix(x1 * s), -pbar, g, h, a, b)['x']
    return np.array(wt)[:, 0], (returns, vols)


class MinVolPortfolio(InvVolRiskParity):
    """Minimum Volatility optimal portfolio.

    Input time-series Data Frame with columns as tickers
    """

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.frontier = None
        return

    @property
    def data(self):
        return self._data

    @property
    def symbols(self):
        return self._data.columns.tolist()

    @property
    def weight(self):
        return self._weight

    def run(self):
        # cannot use close-form solution, it can yield negative weights
        cov = self.data.cov()
        wt, (rets, vols) = get_min_var_portfolio(cov.values)
        self._weight = pd.Series(wt, index=cov.columns)
        self.frontier = (rets, vols)
        return

