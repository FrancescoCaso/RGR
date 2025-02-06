import os
import time
import torch
import numpy as np
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Airports
import torch_geometric
from LaplacianRenormalizationGroup4 import compute_trace_rho_L, compute_C_from_file, plot_C_vs_tau

#--------------------------------------
# SET ALL POSSIBLE SEEDS
#--------------------------------------
seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Definizione dei dataset e dei tau
# datasets = ['cora', 'citeseer', 'pubmed', 'Computers', 'Photo', 'Europe']
datasets = ['Computers']
epsilon = 0.05
# Creazione di un array in scala logaritmica tra 10^-2 e 10^4
start = -2  # Esponente iniziale
end = 4     # Esponente finale
num_points = 100  # Numero di punti nell'array
tau_values= np.logspace(start, end, num=num_points)

# Directory principale per i dati
cwd = os.getcwd()
data_dir0 = os.path.join(cwd, 'RGNN_data4')
if not os.path.exists(data_dir0):
    os.mkdir(data_dir0)

# Trasformazione per ottenere la componente più grande
transform = T.Compose([T.LargestConnectedComponents()])

for d in datasets:
    t1 = time.time()
    data_dir1 = os.path.join(data_dir0, d)
    if not os.path.exists(data_dir1):
        os.mkdir(data_dir1)
    vanilla_dir = os.path.join(data_dir1, 'vanilla')
    ren_dir = os.path.join(data_dir1, 'renormalized')

    if d == 'cora' or d == 'citeseer' or d == 'pubmed':
        temp_dataset = Planetoid(root="", name=d, transform=transform)
    elif d == 'Computers' or d == 'Photo':
        temp_dataset = Amazon(root='', name=d, transform=transform)
    elif d == 'Europe':
        temp_dataset = Airports(root='', name=d, transform=transform)
    else:
        raise ValueError

    for ind, data in enumerate(temp_dataset):
        t1 = time.time()
        edge_index0 = data.edge_index
        A = torch_geometric.utils.to_dense_adj(edge_index0)

        # Step 1: Calcola Trace[rho*L] per diversi valori di tau e salva in un file txt
        trace_file = os.path.join(data_dir1, f'{d}_data_{ind}_trace.txt')
        compute_trace_rho_L(A, tau_values, trace_file)

        # Step 2: Calcola l'array C e salva in un file txt
        C_file = os.path.join(data_dir1, f'{d}_data_{ind}_C.txt')
        tau_array, C_array = compute_C_from_file(trace_file, C_file)

        # Step 3: Genera e salva un grafico di C rispetto a tau
        plot_image_file = os.path.join(data_dir1, f'{d}_data_{ind}_C_vs_tau.png')
        plot_C_vs_tau(tau_array, C_array, plot_image_file)
    t2=time.time()

    print(f'Processed dataset {d}: {len(temp_dataset)} graphs')
    print(t2-t1)