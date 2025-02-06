from LaplacianRenormalizationGroup4 import *
import torch_geometric
import os
import torch
import numpy as np
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Airports

#--------------------------------------
# SET ALL POSSIBLE SEEDS
#--------------------------------------
seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

epsilon = 0.05

kind = "maximum"
dataset = ['cora', 'citeseer', 'Europe', 'pubmed', 'Computers', 'Photo']

cwd = os.getcwd()
print(cwd)

transform = T.Compose([T.LargestConnectedComponents()])

data_dir0 = os.path.join(cwd, 'RGNN_data4')
if not os.path.exists(data_dir0):
    os.mkdir(data_dir0)

for d in dataset:
    data_dir1 = os.path.join(data_dir0, d)
    if not os.path.exists(data_dir1):
        os.mkdir(data_dir1)

    vanilla_dir = os.path.join(data_dir1, 'vanilla')
    if not os.path.exists(vanilla_dir):
        os.mkdir(vanilla_dir)

    ren_dir = os.path.join(data_dir1, 'renormalized')
    if not os.path.exists(ren_dir):
        os.mkdir(ren_dir)

    if d == 'cora' or d == 'citeseer' or d == 'pubmed': temp_dataset = Planetoid(root="", name=d, transform=transform)
    elif d == 'Computers' or d == 'Photo': temp_dataset = Amazon(root='', name=d, transform=transform)
    elif d == 'Brazil' or d == 'Europe' or d == 'USA': temp_dataset = Airports(root='', name=d, transform=transform)
    else:
        raise ValueError

    for ind1, data in enumerate(temp_dataset):
         torch.save(data,
                       os.path.join(vanilla_dir,
                                    f'data_{ind1}.pt'))  

    tau = extract_tau_for_type(f"{data_dir1}/{d}_data_0_tau_type.txt", kind=kind) #for real characteristics

    for t in tau:
    # t = tau[0]
        print(f'tau: {t}')
        r_dir2 = os.path.join(ren_dir, f't_{t}')
        if not os.path.exists(r_dir2):
                os.mkdir(r_dir2)
        for ind, data in enumerate(temp_dataset):
            
            edge_index0 = data.edge_index
            A = torch_geometric.utils.to_dense_adj(edge_index0)

            CG_adj, prop_adj = RSLRG(A, t)

            ren_prop_adj = ((prop_adj - CG_adj) > epsilon).int()
            
            prop_edge_index = torch_geometric.utils.dense_to_sparse(ren_prop_adj)[0]
            print(prop_edge_index.shape)

            
            data.edge_index = prop_edge_index

            torch.save(data,
                        os.path.join(r_dir2,
                                    f'data_{ind}.pt'))
           
            print(f'processed at t {t}: {ind + 1}')