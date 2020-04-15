"""
Classes for all models and loss functions for clustering.

FC --> ReLU --> BN --> Dropout --> FC
"""

import pdb
import warnings
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import utils
import config
import lorentz


def sqeuclidean_pdist(x, y=None):
    """Fast and efficient implementation of ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2 X^T Y
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """

    x_norm = (x**2).sum(1).unsqueeze(1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).unsqueeze(0)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.squeeze().unsqueeze(0)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # get rid of NaNs
    dist[torch.isnan(dist)] = 0.
    # clamp negative stuff to 0
    dist = torch.clamp(dist, 0., np.inf)
    # ensure diagonal is 0
    if y is None:
        dist[dist == torch.diag(dist)] = 0.

    return dist

# ============================================================================ #
#                                LOSS FUNCTIONS                                #
# ============================================================================ #

class BallClusterLearningLoss(nn.Module):
    """Final BCL method
    space: 'sqeuclidean' or 'lorentz'
    init_bias: initialize bias to this value
    temperature: sampling temperature (decayed in main training loop)
    beta: Lorentz beta for comparison in Lorentz space
    """

    def __init__(self, device, space='sqeuclidean', l2norm=True, gamma_eps=0.05,
                 init_bias=0.1, learn_bias=True, beta=0.01, alpha_pos=4., alpha_neg=1., mult_bias=0.):
        """Initialize
        """
        super(BallClusterLearningLoss, self).__init__()
        self.device = device
        self.space = space
        self.learn_bias = learn_bias
        self.l2norm = l2norm
        self.beta = beta
        self.gamma_eps = gamma_eps
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.mult_bias = mult_bias

        self.h_bias = nn.Parameter(torch.tensor(init_bias))
        self.bias = F.softplus(self.h_bias)

    def forward(self, Xemb, scores, labels):
        """
        Xemb: N x D, N features, D embedding dimension
        labels: ground-truth cluster indices
        NOTE: labels are not necessarily ordered indices, just unique ones, don't use for indexing!
        """

        self.bias = F.softplus(self.h_bias)

        # get unique labels to loop over clusters
        unique_labels = labels.unique()  # torch vector on cuda
        K = unique_labels.numel()
        N = Xemb.size(0)

        # collect centroids, cluster-assignment matrix, and positive cluster index
        centroids = []
        pos_idx = -1 * torch.ones_like(labels)  # N vector, each in [0 .. K-1]
        clst_assignments = torch.zeros(N, K).to(self.device)  # NxK {0, 1} matrix
        for k, clid in enumerate(unique_labels):
            idx = labels == clid
            # assign all samples with cluster clid as k
            pos_idx[idx] = k
            clst_assignments[idx, k] = 1
            # collect all features
            Xclst = Xemb[idx, :]
            centroid = Xclst.mean(0)
            centroid = centroid / centroid.norm()
            # collect centroids
            centroids.append(centroid)
        centroids = torch.stack(centroids, dim=0)

        # pairwise distances between all embeddings of the batch and the centroids
        XC_dist = (Xemb.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(2)

        # add bias to the distances indexed appropriately
        pos_bias = self.bias
        neg_bias = 9 * self.bias + self.gamma_eps

        # add bias and use "cross-entropy" loss on pos_idx
        bias_adds = clst_assignments * pos_bias + (1 - clst_assignments) * neg_bias
        final_distance = (-XC_dist + bias_adds) * 0.1
        # when not using bias, just ignore
        if self.bias == 0.:
            final_distance = -XC_dist * 0.1

        # make sure positive distances are below the pos-bias
        pos_distances = XC_dist.gather(1, pos_idx.unsqueeze(1))
        pos_sample_loss = F.relu(pos_distances - pos_bias)

        # make sure negative distances are more than neg-bias
        #avg_neg_distances = XC_dist[1 - clst_assignments.byte()].view(N, K-1).mean(1)
        min_neg_distances = XC_dist[1 - clst_assignments.byte()].view(N, K-1).min(1)[0]  # [0] returns values not indices
        neg_sample_loss = F.relu(neg_bias - min_neg_distances)

        pos_loss = pos_sample_loss.mean()
        neg_loss = neg_sample_loss.mean()

        losses = {'ctrd_pos': pos_loss * self.alpha_pos,
                  'ctrd_neg': neg_loss * self.alpha_neg}

        return losses


class PrototypicalLoss(nn.Module):
    """Prototypical network like loss with bias
    p_ik = exp(- d(x^k_i, c^k) + b) / (exp(- d(x^k_i, c^k) + b) + sum_j exp(- d(x^k_i, c^j) + 2b))
    Loss = -mean_k( mean_i ( -log p_ik ))
    space: 'sqeuclidean' or 'lorentz'
    init_bias: initialize bias to this value
    temperature: sampling temperature (decayed in main training loop)
    beta: Lorentz beta for comparison in Lorentz space
    """

    def __init__(self, device, space='sqeuclidean', l2norm=False, gamma_eps=0.05,
                 init_bias=0., learn_bias=False, beta=0.01, alpha_pos=1., alpha_neg=1., mult_bias=0.):
        """Initialize
        """
        super(PrototypicalLoss, self).__init__()
        self.device = device
        self.space = space
        self.learn_bias = learn_bias
        self.l2norm = l2norm
        self.beta = beta
        self.gamma_eps = gamma_eps
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.mult_bias = mult_bias

        self.bias = torch.tensor(init_bias).to(self.device)

    def forward(self, Xemb, scores, labels):
        """
        Xemb: N x D, N features, D embedding dimension
        labels: ground-truth cluster indices
        NOTE: labels are not necessarily ordered indices, just unique ones, don't use for indexing!
        """

        unique_labels = labels.unique()  # torch vector on cuda
        K = unique_labels.numel()
        N = Xemb.size(0)

        # collect centroids, cluster-assignment matrix, and positive cluster index
        centroids = []
        pos_idx = -1 * torch.ones_like(labels)  # N vector, each in [0 .. K-1]
        clst_assignments = torch.zeros(N, K).to(self.device)  # NxK {0, 1} matrix
        for k, clid in enumerate(unique_labels):
            idx = labels == clid
            # assign all samples with cluster clid as k
            pos_idx[idx] = k
            clst_assignments[idx, k] = 1
            # collect all features
            Xclst = Xemb[idx, :]
            centroid = Xclst.mean(0)
            # collect centroids
            centroids.append(centroid)
        centroids = torch.stack(centroids, dim=0)

        # pairwise distances between all embeddings of the batch and the centroids
        XC_dist = (Xemb.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(2)

        # add bias to the distances indexed appropriately
        pos_bias = self.bias
        neg_bias = 9 * self.bias + self.gamma_eps
        final_distance = -XC_dist * 0.1

        # compute cross-entropy
        pro_sample_loss = F.cross_entropy(final_distance, pos_idx, reduction='none')

        # do mean of means to get final loss value
        pro_loss = torch.tensor(0.).to(self.device)
        for clid in unique_labels:
            pro_loss += pro_sample_loss[labels == clid].mean()
        pro_loss /= K

        losses = {'ctrd_pro': pro_loss}

        return losses


class ContrastiveLoss(nn.Module):
    """
    In the original paper http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Y = 0 for similar pairs ("positive")
    Y = 1 for dissimilar pairs ("negatives")
    L(Y, X1, X2) = (1 - Y) * 0.5 * D^2 + Y * 0.5 * (max(0, m - D))^2
    NOTE: distance is in Euclidean space, not sqeuclidean!
    """

    def __init__(self, device, l2norm=True,
                 init_bias=1., learn_bias=True):
        """Initialize
        """
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.learn_bias = learn_bias
        self.l2norm = l2norm

        self.h_bias = nn.Parameter(torch.tensor(init_bias))
        self.bias = F.softplus(self.h_bias)

    def forward(self, Xemb, scores, labels):
        """
        Xemb: N x D, N features, D embedding dimension
        labels: ground-truth cluster indices
        """

        self.bias = F.softplus(self.h_bias)

        N = Xemb.size(0)
        match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # a NxN {0,1} matrix

        ### generate positive pairs, and pull corresponding features
        diag_mask = 1 - torch.eye(N).to(self.device)
        pos_idx = (diag_mask * match).nonzero()
        X1_pos = Xemb.index_select(0, pos_idx[:, 0])
        X2_pos = Xemb.index_select(0, pos_idx[:, 1])

        ### generate random negatives
        neg_idx = []
        while len(neg_idx) < X1_pos.size(0):  # match pairs for negatives
            idx = torch.randint(N, (2,)).long()
            if match[idx[0], idx[1]] == 0:
                neg_idx.append(idx)
        neg_idx = torch.stack(neg_idx).to(self.device)

        X1_neg = Xemb.index_select(0, neg_idx[:, 0])
        X2_neg = Xemb.index_select(0, neg_idx[:, 1])

        # compute distances (Euclidean!)
        pos_distances_sq = ((X1_pos - X2_pos) ** 2).sum(1)
        neg_distances = ((X1_neg - X2_neg) ** 2).sum(1).sqrt()

        # Loss = 0.5 * pos_distances_sq   +   0.5 * (max(0, m - neg_distances))^2
        pos_loss = 0.5 * pos_distances_sq.mean()
        neg_loss = 0.5 * (F.relu(self.bias - neg_distances) ** 2).mean()

        return {'cont_pos': pos_loss, 'cont_neg': neg_loss}


class TripletLoss(nn.Module):
    """
    In the FaceNet paper https://arxiv.org/pdf/1503.03832.pdf
    L = max(0, d+ - d- + alpha)
    NOTE: distance is in sqeuclidean space!
    """

    def __init__(self, device, space='sqeuclidean', l2norm=True,
                 init_bias=0.5, learn_bias=False):
        """Initialize
        """
        super(TripletLoss, self).__init__()
        self.device = device
        self.space = space
        self.learn_bias = learn_bias
        self.l2norm = l2norm

        self.bias = torch.tensor(init_bias).to(self.device)

    def forward(self, Xemb, scores, labels):
        """
        Xemb: N x D, N features, D embedding dimension
        labels: ground-truth cluster indices
        """

        N = Xemb.size(0)
        match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # a NxN {0,1} matrix

        ### generate positive pairs, and pull corresponding features
        diag_mask = 1 - torch.eye(N).to(self.device)
        pos_idx = (diag_mask * match).nonzero()
        anc_idx = pos_idx[:, 0]
        pos_idx = pos_idx[:, 1]

        ### generate negatives for the same anchors as positive
        neg_idx = torch.zeros_like(pos_idx).long()
        for k in range(pos_idx.size(0)):
            this_negs = torch.nonzero(1 - match[pos_idx[k]]).squeeze()
            neg_idx[k] = this_negs[torch.randperm(this_negs.size(0))][0]

        X_anc = Xemb.index_select(0, anc_idx)
        X_pos = Xemb.index_select(0, pos_idx)
        X_neg = Xemb.index_select(0, neg_idx)

        # compute distances
        pos_distances = ((X_anc - X_pos) ** 2).sum(1)
        neg_distances = ((X_anc - X_neg) ** 2).sum(1)

        # loss
        loss = F.relu(self.bias + pos_distances - neg_distances).mean()

        return {'trip': loss}


class LogisticDiscriminantLoss(nn.Module):
    """Pairwise distance between samples, using logistic regression
    https://hal.inria.fr/file/index/docid/439290/filename/GVS09.pdf
    space: 'sqeuclidean' or 'lorentz'
    init_bias: initialize bias to this value (or as set by radius)
    temperature: sampling temperature (decayed in main training loop)
    with_ball: loss being used along with ball loss?
    beta: Lorentz beta for comparison in Lorentz space
    """

    def __init__(self, device, space='sqeuclidean',
                 init_bias=0.5, learn_bias=True, temperature=1., beta=0.01,
                 with_ball=False):
        """Initialize
        """
        super(LogisticDiscriminantLoss, self).__init__()
        self.device = device
        self.space = space
        self.temperature = temperature
        
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, Xemb, scores, labels):
        """
        Xemb: N x D, N features, D embedding dimension
        labels: ground-truth cluster indices
        """

        N = Xemb.size(0)
        match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # a NxN {0,1} matrix

        ### generate positive pairs, and pull corresponding features
        diag_mask = 1 - torch.eye(N).to(self.device)
        pos_idx = (diag_mask * match).nonzero()
        X1_pos = Xemb.index_select(0, pos_idx[:, 0])
        X2_pos = Xemb.index_select(0, pos_idx[:, 1])

        ### generate random negatives
        neg_idx = []
        while len(neg_idx) < X1_pos.size(0):  # match pairs for negatives
            idx = torch.randint(N, (2,)).long()
            if match[idx[0], idx[1]] == 0:
                neg_idx.append(idx)
        neg_idx = torch.stack(neg_idx).to(self.device)

        X1_neg = Xemb.index_select(0, neg_idx[:, 0])
        X2_neg = Xemb.index_select(0, neg_idx[:, 1])

        # compute distances
        pos_distances = ((X1_pos - X2_pos) ** 2).sum(1)
        neg_distances = ((X1_neg - X2_neg) ** 2).sum(1)

        # Loss = -y log(p) - (1-y) log(1-p)
        pos_logprobs = torch.sigmoid((self.bias - pos_distances)/self.temperature)
        neg_logprobs = torch.sigmoid((self.bias - neg_distances)/self.temperature)
        pos_loss = -(pos_logprobs).log().mean()
        neg_loss = -(1 - neg_logprobs).log().mean()

        return {'ldml_pos': pos_loss, 'ldml_neg': neg_loss}


