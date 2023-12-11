# quantized-gnn-nas (GraQNAS)

This work is on quantization for GNN-based NAS. 
Actively under development by @nargesalavi and MohamadAli Seyed Mahmoud

AutoGL is developed for researchers and developers to conduct quantization on neural architecture search for graph datasets and tasks easily and quickly. 

## Running GraQNAS

The notebook in quantized-gnn-nas/graph_nas.ipynb contains the required packages and experiments that are needed for running AutoGEL and finding its best test accuracy along with its number of parameters and their required memory compared with the ones needed for GraQNAS.


Use the following command to change directories:

`cd AutoGEL/nc_gc/nc_gc_darts/`

To run a node classification task, set --task to 'node' and specify the dataset with --dataset. For example, to perform node classification on the Cora dataset, use the following command:

`!python main.py --task 'node' --dataset 'Cora'`

Similarly, for graph classification tasks, set --task to 'graph' and choose your desired dataset. For instance, to run graph classification on the IMDB-MULTI dataset, the command would be:

`!python main.py --task 'graph' --dataset 'IMDB-MULTI'`




