import torch
import torch.nn as nn
from genotypes import GNN_OPS
from operations import *

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in GNN_OPS:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class GNNCell(nn.Module):
    # This will define the GNN Cell using the MixedOp
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, genotype):
        # Initialization code...

    def forward(self, s0, s1, weights):
        # Forward propagation code...


class GNNSearch(nn.Module):
    def __init__(self, C, num_classes, num_cells, genotype):
        # Initialization code...

    def forward(self, input):
        # Forward propagation code...
