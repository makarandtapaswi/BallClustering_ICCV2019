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


# ============================================================================ #
#                                   MODELS                                     #
# ============================================================================ #

class ResNetLinearBlock(nn.Module):
    # NOT USED
    pass


class EmbedMLP(nn.Module):
    def __init__(self, mlp_dims, nonlin='relu', use_bn=False,
                 l2norm=False, dropout=0.0, resnet_blocks=False,
                 use_classifier=0):
        """Define MLP with standard regularization features
        mlp_dims: a list of integers, creates layers based on this
        nonlin: nonlinearity (will choose this function to apply everywhere)
        l2norm: perform l2-normalization on embedding (output) features?
        use_bn: batch-norm at every layer?
        dropout: dropout of x at every layer?
        """

        super(EmbedMLP, self).__init__()
        # l2norm?
        self.l2norm = l2norm

        ### nonlinearity ###
        if nonlin == 'relu':
            self.nonlin = F.relu
        elif nonlin == 'prelu':
            self.nonlin = F.prelu
        elif nonlin == 'tanh':
            self.nonlin = torch.tanh

        ### MLP with N linear / resnet block layers ###
        self.linears = nn.ModuleList()
        for k in range(1, len(mlp_dims)):
            # swap between standard linear layer and ResNet blocks
            if resnet_blocks and mlp_dims[k-1] == mlp_dims[k]:
                self.linears.append(ResNetLinearBlock(mlp_dims[k], self.nonlin))
            else:
                self.linears.append(nn.Linear(mlp_dims[k-1], mlp_dims[k]))

        ### stick a classifier at the top? ###
        self.use_classifier = use_classifier > 0
        if self.use_classifier:
            self.classifier = nn.Linear(mlp_dims[-1], use_classifier)

        ### batch-norms ###
        self.use_bn = use_bn
        if use_bn:
            self.batch_norms = nn.ModuleList()
            for k in range(1, len(mlp_dims)):
                self.batch_norms.append(nn.BatchNorm1d(mlp_dims[k]))

        ### reset weights? ###
        # for layer in self.linears:
        #     nn.init.uniform_(layer.weight, -1e-4, 1e-4)
        #     nn.init.uniform_(layer.bias,   -1e-4, 1e-4)

        ### dropout ###
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # X: BS x D

        # MLP
        for k, layer in enumerate(self.linears):
            x = layer(x)
            # if last layer, don't do additional processing and other stuff
            if k != (len(self.linears) - 1):
                x = self.nonlin(x)
                if self.use_bn:
                    x = self.batch_norms[k](x)
                if self.use_dropout:
                    x = self.dropout(x)

        # compute scores
        scores = None
        if self.use_classifier:
            scores = self.classifier(self.nonlin(x))

        else:
            # l2 normalize
            if self.l2norm:
                x = x / x.norm(dim=1, keepdim=True)

        return x, scores


# ============================================================================ #
#                                LOSS FUNCTIONS                                #
# ============================================================================ #
## detailed loss functions will be added when the training code is released

class BallCriterion(nn.Module):
    pass

class DeepSpectralClusteringLoss(nn.Module):
    pass

class LogisticDiscriminantLoss(nn.Module):
    pass

class CentroidLoss(nn.Module):
    pass

class ContrastiveLoss(nn.Module):
    pass

class TripletLoss(nn.Module):
    pass

class CrossEntropyLoss(nn.Module):
    pass

class FineTunePairLoss(nn.Module):
    pass

