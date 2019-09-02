import numpy as np


def weighted_purity(Y, C):
    """Computes weighted purity of HAC at one particular clustering "C".
    Y, C: np.array([...]) containing unique cluster indices (need not be same!)
    Note: purity --> 1 as the number of clusters increase, so don't look at this number alone!
    """

    purity = 0.
    uniq_clid, clustering_skew = np.unique(C, return_counts=True)
    num_samples = np.zeros(uniq_clid.shape)
    # loop over all predicted clusters in C, and measure each one's cardinality and purity
    for k in uniq_clid:
        # gt labels for samples in this cluster
        k_gt = Y[np.where(C == k)[0]]
        values, counts = np.unique(k_gt, return_counts=True)
        # technically purity = max(counts) / sum(counts), but in WCP, the sum(counts) multiplies to "weight" the clusters
        purity += max(counts)

    purity /= Y.shape[0]
    return purity, clustering_skew


def NMI(Y, C):
    """Normalized Mutual Information: Clustering performance between ground-truth Y and prediction C
    Based on https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf

    Result matches examples on pdf
    Example:
    Y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2])
    NMI(Y, C) = 0.1089

    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    NMI(Y, C) = 0.2533
    """

    def entropy(labels):
        # H(Y) and H(C)
        H = 0.
        for k in np.unique(labels):
            p = (labels == k).sum() / labels.size
            H -= p * np.log2(p)
        return H

    def h_y_given_c(labels, pred):
        # H(Y | C)
        H = 0.
        for c in np.unique(pred):
            p_c = (pred == c).sum() / pred.size
            labels_c = labels[pred == c]
            for k in np.unique(labels_c):
                p = (labels_c == k).sum() / labels_c.size
                H -= p_c * p * np.log2(p)
        return H

    h_Y = entropy(Y)
    h_C = entropy(C)
    h_Y_C = h_y_given_c(Y, C)
    # I(Y; C) = H(Y) - H(Y|C)
    mi = h_Y - h_Y_C
    # NMI = 2 * MI / (H(Y) + H(C))
    nmi = 2 * mi / (h_Y + h_C)
    return nmi

