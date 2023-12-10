import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, GINConv, global_max_pool

OPS = {
  'gcn': lambda in_feats, out_feats, affine: GCNLayer(in_feats, out_feats, affine),
  'gat': lambda in_feats, out_feats, affine: GATLayer(in_feats, out_feats, affine),
  'gin': lambda in_feats, out_feats, affine: GINLayer(in_feats, out_feats, affine),

  'sage': lambda in_feats, out_feats, affine: SAGELayer(in_feats, out_feats, affine),
  'identity': lambda in_feats, out_feats, affine: Identity(),
  'zero': lambda in_feats, out_feats, affine: Zero(),

   'mean_pool': lambda in_feats, out_feats, affine: GlobalMeanPool(),
   'max_pool': lambda in_feats, out_feats, affine: GlobalMaxPool(),
#    'add': lambda in_feats, out_feats, affine: ElementWiseAdd(),
#    'concat': lambda in_feats, out_feats, affine: FeatureConcat(),
    
}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, affine=True):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_feats, out_feats, improved=affine)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, affine=True, heads=1):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_feats, out_feats, heads=heads, concat=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class GINLayer(nn.Module):
    def __init__(self, in_feats, out_feats, affine=True):
        super(GINLayer, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU(), nn.Linear(out_feats, out_feats))
        self.conv = GINConv(nn1)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class SAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, affine=True):
        super(SAGELayer, self).__init__()
        self.conv = SAGEConv(in_feats, out_feats)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_index=None):
        return x

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x, edge_index=None):
        return x * 0

# Optional: Pooling layers for graph classification tasks
class GlobalMeanPool(nn.Module):
    def __init__(self):
        super(GlobalMeanPool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

# class ElementWiseAdd(nn.Module):
#     def __init__(self):
#         super(ElementWiseAdd, self).__init__()

#     def forward(self, x1, x2):
#         return x1 + x2

# class FeatureConcat(nn.Module):
#     def __init__(self):
#         super(FeatureConcat, self).__init__()

#     def forward(self, x1, x2):
#         return torch.cat((x1, x2), dim=-1)

    


# You can also add custom layers or more sophisticated GNN layers based on your needs.
