import os.path as osp
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d


def RSLRG(adjacency_matrix, tau:float):
    """computes the tau real-space renormalized adjacency matrix

        Args:
            adjacency_matrix: np.array of shape '(num_nodes, num_nodes)'
            tau: float describing the renormalization parameter

        Returns:
            ren_adjacency_matrix: np.array of shape '(num_macro_nodes, num_macro_nodes)'
            where num_macro_nodes is the number of groups of nodes (i.e. macro-nodes)
            macro_nodes: list of macronodes labels

        Requirements:
            numpy as np
            scipy.linalg
        """
    #print('check')
    A = adjacency_matrix[0]

    D = torch.diag(torch.sum(A, dim=0))  # create Degree Matrix of the adjacency matrix
    L = D - A # compute the Laplacian
    rho = torch.linalg.matrix_exp(-tau * L) # compute the rho matrix, i.e. matrix of information flows (non normalized)
    auto_rho = torch.diag(rho) # creates an array of all the auto-information flows (non normalized)
    compar = [torch.min(a, auto_rho) for a in auto_rho] # create matrix of comparison terms necessary for the zeta matrix
    zeta = (rho >= torch.vstack(compar)).int() # computes the zeta matrix associated to the meta-graph

    # maybe the following lines can be improved
    A_new = [A[:, torch.nonzero(z).T].sum(axis=2) for z in zeta] # sums all the incomings edges between nodes in the same group
    A_new = torch.hstack(A_new)
    A_new = [A_new[torch.nonzero(z).T, :].sum(dim=1) for z in zeta] # sums all the outcomings edges between nodes in the same group
    A_new = torch.vstack(A_new) # re-creates the np.array

    # creating the coarse-graning graph
    zeta_new = [zeta[:, torch.nonzero(z).T].sum(axis=2) for z in zeta]
    zeta_new = torch.hstack(zeta_new)
    zeta_new = [zeta_new[torch.nonzero(z).T, :].sum(axis=1) for z in zeta]
    zeta_new = torch.vstack(zeta_new)
    CG_adj = (zeta_new > 0).int()

    return CG_adj, (A_new > 0).int()


def compute_trace_rho_L(adjacency_matrix, tau_list, output_file):
    """Computes the trace of the product of rho and L for different values of tau 
       and writes the results to a text file.

    Args:
        adjacency_matrix: np.array of shape '(num_nodes, num_nodes)'
        tau_list: list of float values for tau
        output_file: str, path to the output file
    """
    # Prepare to write to the output file
    with open(output_file, 'w') as f:
        f.write("tau\tTrace[rho*L]\n")  # Write the header

        for tau in tau_list:
            A = adjacency_matrix[0]

            # Compute Degree Matrix and Laplacian
            D = torch.diag(torch.sum(A, dim=0))  
            L = D - A

            # Compute rho matrix
            rho = torch.linalg.matrix_exp(-tau * L)

            # Compute the trace of rho * L
            trace_rho_L = torch.trace(torch.matmul(rho, L))

            # Write tau and the trace to the file
            f.write(f"{tau}\t{trace_rho_L.item():.6f}\n")

def compute_C_from_file(file_path, output_file):
    """Computes the array C = -tau^2 d(Trace[rho*L])/dtau from the data in the file.
    
    Args:
        file_path: str, path to the file containing tau and Trace[rho*L] values.
    
    Returns:
        tau_array: np.array, array of tau values.
        C_array: np.array, array of C values.
    """
    # Step 1: Read the data from the file
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)  # Skip the header row
    tau_array = data[:, 0]  # First column is tau
    trace_rho_L_array = data[:, 1]  # Second column is Trace[rho*L]
    
    # Step 2: Compute the derivative of Trace[rho*L] with respect to tau
    d_trace_rho_L_dtau = np.gradient(trace_rho_L_array, tau_array)
    
    # Step 3: Compute C = -tau^2 * d(Trace[rho*L])/dtau
    C_array = -tau_array**2 * d_trace_rho_L_dtau
    
    with open(output_file, 'w') as f:
        f.write("tau\tC\n")
        for tau, C in zip(tau_array, C_array):
            f.write(f"{tau}\t{C:.6f}\n")

    return tau_array, C_array

# Funzione per creare un grafico di C rispetto a tau
def plot_C_vs_tau(tau_array, C_array, output_image_file):
    plt.figure(figsize=(10, 6))
    plt.plot(tau_array, C_array, marker='o', linestyle='-', color='b', label='$C = -\\tau^2 \\frac{d(Trace[\\rho L])}{d\\tau}$')
    plt.title('Plot of $C$ vs $\\tau$', fontsize=16)
    plt.xlabel('$\\tau$', fontsize=14)
    plt.ylabel('$C$', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_image_file)
    plt.close()

def find_inflections(y):
    """Find inflection points by checking the sign change of the second derivative."""
    dy = np.gradient(y)
    d2y = np.gradient(dy)
    inflection_points = np.where(np.diff(np.sign(d2y)))[0]
    return inflection_points

def identify_extrema_and_inflections(input_file, output_file):
    # Leggi il file
    data = np.loadtxt(input_file, delimiter='\t', skiprows=1)
    tau_array = data[:, 0]
    C_array = data[:, 1]

    # Smussare i dati per evitare picchi numerici
    C_smooth = gaussian_filter1d(C_array, sigma=1)

    # Trova i massimi locali
    maxima_indices = argrelextrema(C_smooth, np.greater)[0]

    # Trova i punti di flesso
    inflection_indices = find_inflections(C_smooth)

    # Scrivi i risultati su un file di output
    with open(output_file, 'w') as f:
        f.write("tau\ttype\n")
        for i in maxima_indices:
            f.write(f"{tau_array[i]}\tmaximum\n")
        for i in inflection_indices:
            f.write(f"{tau_array[i]}\tinflection\n")

def extract_tau_for_type(input_file, kind):
    """
    Estrae tutti i valori di tau associati al tipo "maximum" o "inflection" da un file di testo.
    
    Args:
        input_file (str): Il percorso del file di input con il formato "tau\ttype".
    
    Returns:
        np.array: Un array di float contenente tutti i valori di tau associati a "maximum" o "inflection".
    """
    # Inizializza una lista per memorizzare i valori di tau associati a "maximum" o "inflection"
    tau_list = []
    if kind != "maximum" and kind != "inflection":
        raise ValueError(f"Error: the value of 'kind' must be either 'maximum' or 'inflection'")

    # Leggi il file e processa le righe
    with open(input_file, 'r') as f:
        header = f.readline()  # Leggi l'intestazione
        for line in f:
            tau, type_str = line.strip().split('\t')
            if type_str == kind:
                tau_list.append(float(tau))

    # Converti la lista in un array numpy
    tau_list_array = np.array(tau_list, dtype=float)
    return tau_list_array

if __name__ == "__main__":
    # from torch_geometric.data import Data
    import torch_geometric
    # import Alon.common
    import os
    import torch
    import numpy as np
    import random
    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset
    from torch_geometric.datasets import Planetoid
    from torch_geometric.datasets import Amazon, Twitch, HeterophilousGraphDataset


    t = 1.0

    # Trasformazione per mantenere i componenti connessi più grandi
    transform = T.Compose([T.LargestConnectedComponents()])

    temp_dataset = TUDataset(root='', name="ENZYMES", transform=transform)


    for ind, data in enumerate(temp_dataset):
        
        edge_index0 = data.edge_index
        A = torch_geometric.utils.to_dense_adj(edge_index0)

        CG_adj, prop_adj = RSLRG(A, t)