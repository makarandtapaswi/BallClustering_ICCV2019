"""
Train model for face clustering
"""

import os
import sys
import pdb
import glob
import time
import warnings
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
torch.manual_seed(1111)

# Local imports
from hac import HAC
import modules
import metrics


# CPU / GPU
device = None


def validate(hac, dset, model, curve=False):
    """Evaluate model performance
    """

    model.eval()
    # evaluation dataset is simple
    X, y_gt = dset

    ### EMBEDDINGS ###
    print('Computing embeddings')
    with torch.no_grad():
        X = torch.from_numpy(X).to(device)
        Xemb, _ = model(X)

    ### CLUSTERING ###
    print('Performing clustering')
    Z, C = hac(Xemb)

    ### CLUSTERING METRICS ###
    nT = C.size  # number of tracks
    # number of clusters
    nY = np.unique(y_gt).size  # numel
    nC = np.unique(C).size  # numel
    # metrics
    nmi = metrics.NMI(y_gt, C)
    wcp = metrics.weighted_purity(y_gt, C)[0]
    # print, store and return
    print('#Clusters in T: {:5d}, Y: {:4d}, C: {:4d}, NMI: {:.4f}, Purity: {:.4f}'.format(nT, nY, nC, nmi, wcp))
    val_metrics = {'nmi': nmi, 'wcp': wcp, 'nY': nY, 'nC': nC}

    # return packaging
    return_things = [val_metrics]
    if curve:
        # purity curve
        curves = hac.evaluate_curve(y_gt, Z, 200, curve_metrics=['wcp', 'nmi'])
        return_things.append(curves)

    return return_things


def simple_read_dataset(video):
    """Simple dataset reading function for purpose of checking evaluation code
    """

    print('Loading dataset:', video)

    # Read label file
    label_fname = 'data/ids/' + video + '.ids'
    with open(label_fname, 'r') as fid:
        fid.readline()  # ignore header
        data = fid.readlines()  # track to name
        data = [line.strip().split() for line in data if line.strip()]
        # trackid --> name mapping
        ids = {int(line[0]): line[1] for line in data}

    # get unique names and assign numbers
    uniq_names = list(set(ids.values()))

    # Read feature files
    X, y = [], []
    all_feature_fname = glob.glob('data/features/' + video + '/*.npy')
    for fname in all_feature_fname:
        # load and append feature
        feat = np.load(fname)
        X.append(feat.mean(0))
        # append label
        tid = int(os.path.splitext(os.path.basename(fname))[0])
        y.append(uniq_names.index(ids[tid]))

    X = np.array(X)
    y = np.array(y)
    return [X, y]


def main(video):
    """Main function
    """

    ### Arguments used during training -- removed the args manager for simplicity during evaluation
    # --dspace sqeuclidean
    # --init_ctrdbias 0.1
    # --loss_components ctrd_pos ctrd_neg
    # --mlp_dims 256 128 64 64
    # --l2norm
    # --learn_ctrdbias
    # --critparam_train_epoch 0
    # --batch_size 2000
    # --ctrd_alpha_pos 4
    # --ctrd_alpha_neg 1
    # --gamma_eps 0.05

    gpu = 0
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu != -1 else "cpu")
    print(device)

    ### Dataset ###
    # simplified evaluation example (normally uses PyTorch datasets)
    X, y = simple_read_dataset(video)

    ### Create Model ###
    model = modules.EmbedMLP(mlp_dims=[256, 256, 128, 64, 64], nonlin='relu', use_bn=False, l2norm=True,
                             dropout=False, resnet_blocks=False, use_classifier=False)
    model = model.to(device)
    print(model)

    ### Load checkpoint ###
    print('Loading checkpoint')
    chkpt_fname = 'model_chkpts/20181113-235913.mlp-256-256-128-64-64.ep-0127.trn-03.5879.bs-2000.pth.tar'
    checkpoint = torch.load(chkpt_fname)
    model.load_state_dict(checkpoint['model_state'])

    ### HAC ###
    hac = HAC(stop_criterion='distance',
              distance_metric='sqeuclidean',
              linkage_method='complete')
    # set the HAC threshold to be 4*b!
    # IMPORTANT: the threshold is learned as part of the criterion module, and not the main MLP model
    hac.thresh = 4 * F.softplus(checkpoint['criterion_state']['ctrd.h_bias']).item()

    ### Run evaluation ###
    val_metrics = validate(hac, [X, y], model, curve=False)


valid_videos = ['bbt_s01e01', 'bbt_s01e02', 'bbt_s01e03', 'bbt_s01e04', 'bbt_s01e05', 'bbt_s01e06',
                'buffy_s05e01', 'buffy_s05e02', 'buffy_s05e03', 'buffy_s05e04', 'buffy_s05e05', 'buffy_s05e06']

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1]:
        video = sys.argv[1]
    else:
        video = 'bbt_s01e01'
    assert video in valid_videos, 'Erroneous video name. Valid videos: {}'.format(valid_videos)

    main(video)
