"""
HAC
"""

import pdb
import numpy as np
import collections
from scipy.cluster import hierarchy as sphac
from scipy.spatial import distance as spdist

# Local imports
import metrics

class HAC(object):
    """Agglomerative clustering.
    """

    def __init__(self, threshold=1., stop_criterion='distance', distance_metric='sqeuclidean', linkage_method='complete'):
        """Default to complete/max linkage, seems to work as well as ward, and has margin-guarantees at test time, allowing to stop easily
        """
        self.thresh = threshold
        self.crit = stop_criterion
        self.metric = distance_metric
        self.link = linkage_method
        self.lorentz_beta = None  # only applicable if using distance_metric == 'lorentz'

    def __call__(self, X=None, Z=None, C=None, neg_pairs=None):
        """Hierarchical Agglomerative Clustering on data X: NxD
        Returns linkage and cluster assignment.

        See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        """

        # Z = sphac.linkage(X, method=self.link, metric=self.metric)
        if self.metric in ['euclidean', 'sqeuclidean']:
            np_X = X.cpu().detach().numpy()
            np_D = spdist.pdist(np_X, metric=self.metric)

        # elif self.metric == 'lorentz':
        #     D = lorentz.pdist(A=X, B=torch.tensor(X), beta=self.lorentz_beta, same=True)
        #     np_D = spdist.squareform(D.cpu().detach().numpy())

        # set distances within negative pairs to big!
        if neg_pairs is not None:
            max_D = max(np_D) * 2
            square_D = spdist.squareform(np_D)
            neg_pairs = np.array(neg_pairs)
            square_D[neg_pairs[:, 0], neg_pairs[:, 1]] = max_D
            np_D = spdist.squareform(square_D)

        Z = sphac.linkage(np_D, metric=self.metric, method=self.link)
        C = sphac.fcluster(Z, self.thresh, criterion=self.crit)
        return Z, C

    def evaluate_curve(self, y, Z, N, t=1, curve_metrics=['wcp']):
        """Evaluate based on dendrogram, from start to end.
        Cut to create clusters at every step from 1:t:N
        """

        wcp_curve = collections.OrderedDict()  # list of tuples of #clusters --> metric
        nmi_curve = collections.OrderedDict()
        for k in range(1, N+1, t):
            C = sphac.fcluster(Z, k, criterion='maxclust')
            # need not actually result in "k" clusters :/
            nc = np.unique(C).shape[0]
            wcp_curve[nc] = metrics.weighted_purity(y, C)[0]
            nmi_curve[nc] = metrics.NMI(y, C)

        # convert to list of tuples
        wcp_curve = [(nc, pur) for nc, pur in wcp_curve.items()]
        nmi_curve = [(nc, nmi) for nc, nmi in nmi_curve.items()]

        if 'wcp' in curve_metrics and 'nmi' in curve_metrics:
            return [wcp_curve, nmi_curve]
        elif 'wcp' in curve_metrics:
            return wcp_curve
        elif 'nmi' in curve_metrics:
            return nmi_curve
