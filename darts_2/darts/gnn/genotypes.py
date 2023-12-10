from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',                     # No operation
    'gcn',                      # Graph Convolutional Network operation
    'gat',                      # Graph Attention Network operation
    'gin',                      # Graph Isomorphism Network operation
    'sage',                     # GraphSAGE operation
    'mean_pool',                # Mean pooling
    'max_pool',                 # Max pooling
    'add',                      # Element-wise addition
    'concat'                    # Feature concatenation
    # More operations can be added as required
]

# Let's define a hypothetical genotype for a GNN architecture
GNN_Net = Genotype(
  normal = [
    ('gat', 1),
    ('gcn', 0),
    ('gin', 0),
    ('gcn', 0),
    ('mean_pool', 1),
    ('gat', 0),
    ('mean_pool', 0),
    ('mean_pool', 0),
    ('gcn', 1),
    ('gat', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('gcn', 1),
    ('gin', 0),
    ('max_pool', 1),
    ('gin', 0),
    ('mean_pool', 1),
    ('gat', 0),
    ('add', 3),
    ('mean_pool', 2),
    ('gcn', 2),
    ('max_pool', 1),
  ],
  reduce_concat = [4, 5, 6]
)

# The above is just an example, you'll need to tailor it to the needs of your GNN architecture.

