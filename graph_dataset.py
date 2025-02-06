import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Dataset, Data
import networkx as nx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, to_networkx
from typing import List

def set_train_val_test_split(
        seed: int,
        data: Data,
        train_percentage: float = 0.6,
        test_percentage: float = 0.1,
        val_percentage: float = 0.1) -> Data:

    assert train_percentage + test_percentage + val_percentage <= 1, "The sum of percentages should not exceed 100%"

    rnd_state = np.random.RandomState(42)
    device = data.y.device
    num_nodes = data.y.shape[0]

    # Split indices into development (train + val) and test sets
    num_test = int(num_nodes * test_percentage)
    test_idx = rnd_state.choice(num_nodes, num_test, replace=False)
    development_idx = [i for i in np.arange(num_nodes) if i not in test_idx]

    # Split development indices into train and validation sets
    num_train = int(len(development_idx) * train_percentage)
    num_val = len(development_idx) - num_train
    rnd_state.shuffle(development_idx)
    train_idx = development_idx[:num_train]
    val_idx = development_idx[num_train:num_train+num_val]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx).to(device)
    data.val_mask = get_mask(val_idx).to(device)
    data.test_mask = get_mask(test_idx).to(device)

    assert data.train_mask.shape[0] == num_nodes, "train_mask size mismatch"
    assert data.val_mask.shape[0] == num_nodes, "val_mask size mismatch"
    assert data.test_mask.shape[0] == num_nodes, "test_mask size mismatch"

    return data



def undirect_connected_component2(dataset):
    # Convert graph to undirected
    dataset.edge_index = to_undirected(dataset.edge_index)

    
    return dataset



def create_masks(data, train_ratio=0.7, val_ratio=0.15):
    num_nodes = data.num_nodes
    device = data.x.device

    # Make sure the ratios sum up to 1
    assert train_ratio + val_ratio <= 1.0, "Ratios must sum to 1 or less."

    # Compute the number of nodes in each split
    num_train = int(train_ratio * num_nodes)
    num_val = int(val_ratio * num_nodes)
    num_test = num_nodes - num_train - num_val

    # Create a list of indices and shuffle them
    indices = torch.arange(num_nodes).to(device)

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Assign True to the indices that belong to each split
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:(num_train + num_val)]] = True
    test_mask[indices[(num_train + num_val):]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def undirect_connected_component(dataset, make_undirected=True, keep_largest_cc=True):
    if make_undirected:
        # Convert graph to undirected
        dataset.edge_index = to_undirected(dataset.edge_index)
    if keep_largest_cc:
        # Convert PyG dataset to NetworkX graph
        G = to_networkx(dataset, to_undirected=True)

        # Get the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)

        # Create a mask for the nodes in the largest connected component
        mask = torch.tensor([node in largest_cc for node in range(dataset.num_nodes)], dtype=torch.bool).to(dataset.edge_index.device)

        # Create an edge mask for the edges in the largest connected component
        edge_mask = mask[dataset.edge_index[0]] & mask[dataset.edge_index[1]]

        # Apply the masks to the node and edge attributes
        dataset.x = dataset.x[mask]  # Node feature matrix
        dataset.y = dataset.y[mask]  # Node labels
        dataset.edge_index = dataset.edge_index[:, edge_mask]  # Edge indices

        # Create masks for train, validation and test sets before applying the mask of the largest connected component
        dataset.train_mask = dataset.train_mask.clone()
        dataset.val_mask = dataset.val_mask.clone()
        dataset.test_mask = dataset.test_mask.clone()

        dataset.train_mask[~mask] = False  # Training set mask
        dataset.val_mask[~mask] = False  # Validation set mask
        dataset.test_mask[~mask] = False  # Test set mask

        # Re-index edge_index to match with the new node index
        _, inverse_indices = torch.unique(dataset.edge_index, return_inverse=True)
        dataset.edge_index = inverse_indices.view_as(dataset.edge_index)
    
    return dataset



class CustomGraphDataset(Dataset):
    def __init__(self, root: str, device: str , transform=None, pre_transform=None, dataset_name: str = "", t: float = 0.0):

        if t == 0.0:
            root = f"{root}/{dataset_name}/vanilla/"
        else:
            root = f"{root}/{dataset_name}/renormalized/t_{t}/"
        self.n_files: int = self._count_files(root)
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.device = device
        self.t = t
        self.dataset_name = dataset_name
        self._load_dataset()

        self.set_undirected = False
        self.set_split = False


    def _load_dataset(self) -> None:
        self.data = [torch.load(f'{self.raw_paths[idx].replace("/raw/", "/")}', map_location=self.device) for idx in range(self.n_files)]
        if isinstance(self.data[0], tuple):
            self.data = self.data[0]

        
        
    def _count_files(self, root: str) -> int:
        return len(os.listdir(root))

    @property
    def raw_file_names(self):
        # Assuming the files are named as 'data_0.pt', 'data_1.pt', etc.
        return [f'data_{i}.pt' for i in range(self.n_files)]

    @property
    def processed_file_names(self):
        # This function is needed for the interface, but not used here
        pass

    def len(self):
        return 1

    def get(self, idx: int) -> Data:
        samples = self.data[0]
        print("non sottografo", samples)

        if self.t in [99, 100]:
            samples = undirect_connected_component(samples, make_undirected=True, keep_largest_cc=False)
            print("UNDIRECTED")

        if not hasattr(self, "train_mask"):
            if self.dataset_name in ["Computers", "Photo", "Europe"]:

                return create_masks(samples)

        
        return samples

    

class CustomMoreGraphDataset(Dataset):
    def __init__(self, root: str, device: str, transform=None, pre_transform=None, dataset_name: str = "", t: float = 0.0, split_ratio: tuple = (0.7, 0.15, 0.15)):
        """
        Dataset per gestire più grafi distinti, con split tra training e test.
        
        Args:
            root (str): Directory principale contenente i dataset.
            device (str): Device su cui caricare i dati (e.g., 'cpu' o 'cuda').
            transform: Trasformazioni da applicare sui dati.
            pre_transform: Trasformazioni da applicare prima delle operazioni principali.
            dataset_name (str): Nome del dataset.
            t (float): Parametro di rinormalizzazione t.
            split_ratio (tuple): Tuple che indica la proporzione del training, validation e test set (es. (0.7, 0.15, 0.15)).
        """
        if t == 0.0:
            root = f"{root}/{dataset_name}/vanilla/"
        else:
            root = f"{root}/{dataset_name}/renormalized/t_{t}/"
        self.n_files: int = self._count_files(root)
        super(CustomMoreGraphDataset, self).__init__(root, transform, pre_transform)
        self.device = device
        self.t = t
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio

        # Carica i grafi
        self._load_dataset()

        # Suddivisione in training e test set
        self.train_graphs, self.val_graphs, self.test_graphs = self._split_dataset()

    def _load_dataset(self) -> None:
        """
        Carica tutti i grafi da file .pt
        """
        # Carica i file dei grafi nella lista self.data
        self.data = [torch.load(f'{self.raw_paths[idx].replace("/raw/", "/")}', map_location=self.device) for idx in range(self.n_files)]
        if isinstance(self.data[0], tuple):
            self.data = [d[0] for d in self.data]  # Gestione delle tuple, se presenti

    def _count_files(self, root: str) -> int:
        """
        Conta il numero di file presenti nella directory specificata.
        """
        return len(os.listdir(root))

    def _split_dataset(self):
        """
        Suddivide i grafi tra training e test set, in base a split_ratio.
        """
        n_train = int(self.split_ratio[0] * self.n_files)
        n_val = int(self.split_ratio[1] * self.n_files)
        n_test = int(self.split_ratio[2] * self.n_files)
        assert n_train + n_val + n_test <= self.n_files
        train_graphs = self.data[:n_train]
        val_graphs = self.data[n_train:n_train+n_val]
        test_graphs = self.data[n_train+n_val:]

        return train_graphs, val_graphs, test_graphs

    @property
    def raw_file_names(self):
        """
        Restituisce i nomi dei file di grafi nel dataset.
        """
        return [f'data_{i}.pt' for i in range(self.n_files)]

    @property
    def processed_file_names(self):
        """
        Questa funzione è necessaria per l'interfaccia di Dataset, ma non viene utilizzata qui.
        """
        pass

    def len(self):
        """
        Restituisce il numero totale di grafi nel dataset.
        """
        return len(self.train_graphs) + len(self.val_graphs) + len(self.test_graphs)

    def get(self, idx: int) -> Data:
        """
        Restituisce il grafo corrispondente all'indice specificato.
        
        Se l'indice è inferiore alla lunghezza del training set, restituisce un grafo di training.
        Altrimenti, restituisce un grafo di test.
        """
        if idx < len(self.train_graphs):
            return self.train_graphs[idx]
        elif idx < len(self.train_graphs) + len(self.val_graphs):
            return self.val_graphs[idx - len(self.train_graphs)]
        else:
            return self.test_graphs[idx - len(self.train_graphs) - len(self.val_graphs)]

class CustomMoreScaledGraphDataset(Dataset):
    def __init__(self, root: str, device: str, transform=None, pre_transform=None, dataset_name: str = "", t_list: List[float] = [0.0], split_ratio: tuple = (0.7, 0.15, 0.15)):
        """
        Dataset per gestire più grafi distinti, con split tra training e test.

        Args:
            root (str): Directory principale contenente i dataset.
            device (str): Device su cui caricare i dati (e.g., 'cpu' o 'cuda').
            transform: Trasformazioni da applicare sui dati.
            pre_transform: Trasformazioni da applicare prima delle operazioni principali.
            dataset_name (str): Nome del dataset.
            t_list (List[float]): Lista di parametri di rinormalizzazione t, un valore per ciascun grafo.
            split_ratio (tuple): Tuple che indica la proporzione del training, validation e test set (es. (0.7, 0.15, 0.15)).
        """
        self.device = device
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.t_list = t_list
        self.root = root
        
        # Carica i grafi
        self.data = self._load_dataset()

        # Suddivisione in training, validation e test set
        self.train_graphs, self.val_graphs, self.test_graphs = self._split_dataset()

    def _load_dataset(self) -> List[Data]:
        """
        Carica i grafi da diverse directory, basate sui valori di `t_list`.
        """
        data_list = []
        
        for t in self.t_list:
            # Determina la directory per il valore specifico di t
            if t == 0.0:
                data_dir = f"{self.root}/{self.dataset_name}/vanilla/"
            else:
                data_dir = f"{self.root}/{self.dataset_name}/renormalized/t_{t}/"

            # Conta i file nella directory corrente
            n_files = self._count_files(data_dir)
            print(f"Caricamento {n_files} grafi da {data_dir} per t={t}")

            # Carica ogni file di grafo e lo aggiunge alla lista data_list
            for idx in range(n_files):
                graph_data = torch.load(f'{data_dir}/data_{idx}.pt', map_location=self.device)
                if isinstance(graph_data, tuple):
                    graph_data = graph_data[0]  # Gestione delle tuple, se presenti
                data_list.append(graph_data)

        return data_list

    def _count_files(self, directory: str) -> int:
        """
        Conta il numero di file presenti nella directory specificata.
        """
        return len([f for f in os.listdir(directory) if f.endswith('.pt')])

    def _split_dataset(self):
        """
        Suddivide i grafi tra training, validation e test set, in base a split_ratio.
        """
        n_files = len(self.data)
        n_train = int(self.split_ratio[0] * n_files)
        n_val = int(self.split_ratio[1] * n_files)
        n_test = n_files - n_train - n_val
        
        train_graphs = self.data[:n_train]
        val_graphs = self.data[n_train:n_train+n_val]
        test_graphs = self.data[n_train+n_val:]

        return train_graphs, val_graphs, test_graphs

    @property
    def raw_file_names(self):
        """
        Restituisce i nomi dei file di grafi nel dataset.
        """
        return [f'data_{i}.pt' for i in range(len(self.data))]

    @property
    def processed_file_names(self):
        """
        Questa funzione è necessaria per l'interfaccia di Dataset, ma non viene utilizzata qui.
        """
        pass

    def len(self):
        """
        Restituisce il numero totale di grafi nel dataset.
        """
        return len(self.data)

    def get(self, idx: int) -> Data:
        """
        Restituisce il grafo corrispondente all'indice specificato.

        Se l'indice è inferiore alla lunghezza del training set, restituisce un grafo di training.
        Altrimenti, restituisce un grafo di validation o test.
        """
        if idx < len(self.train_graphs):
            return self.train_graphs[idx]
        elif idx < len(self.train_graphs) + len(self.val_graphs):
            return self.val_graphs[idx - len(self.train_graphs)]
        else:
            return self.test_graphs[idx - len(self.train_graphs) - len(self.val_graphs)]


if __name__ == "__main__":
    path_to_folder: str = "/home/User/projects/RG_GNN/RGNN_data4"
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dataset = CustomGraphDataset(root=path_to_folder, device=device, dataset_name="cora", t=99)[0]
    print(dataset)