#!/usr/bin/python3

from typing import *
import os
import argparse
from tqdm import tqdm
from utils import dump_configuration_new, read_json, seed_everything, write_json
import torch
from torch_geometric.data import Data
from graph_dataset import CustomGraphDataset
from graph_model import MultiGraphModel, GNN_TYPE
import argparse
from typing import List
import torch
from torch_geometric.data import Data


@torch.no_grad()
def evaluate(model: torch.nn.Module, split: str, *datasets: Data) -> Tuple[float, float, torch.Tensor]:
    assert split in ["dev", "test"]

    # La maschera viene presa dal primo dataset fornito
    primary_data = datasets[0]
    mask = primary_data.val_mask if split == "dev" else primary_data.test_mask

    model.eval()

    # Passa tutti i grafi forniti al modello
    logits = model(*datasets)

    # Prevedi le etichette
    pred = logits.argmax(dim=1)  # Ottieni l'indice della classe con la massima probabilità
    correct = pred[mask] == primary_data.y[mask]  # Calcola l'accuratezza
    acc = int(correct.sum()) / int(mask.sum())

    # Calcola la perdita solo per i nodi nel set di valutazione
    loss = loss_fn(logits[mask], primary_data.y[mask]) # loss_fn is not defined nor given!?

    return acc, loss, pred

def train(model: torch.nn.Module, loss_fn, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, *graph_data: Data):
    model.train()  # Imposta il modello in modalità di addestramento
    optimizer.zero_grad()  # Azzera i gradienti accumulati

    with torch.amp.autocast('cuda', enabled=args.use_fp16):
        # Passiamo tutti i grafi al modello
        out = model(*graph_data)

        # Controlliamo se il primo grafo ha la maschera di training
        primary_data = graph_data[0]  # Il primo grafo sarà quello principale
        if hasattr(primary_data, "train_mask"):
            mask = primary_data.train_mask
            loss = loss_fn(out[mask], primary_data.y[mask])
        else:
            loss = loss_fn(out, primary_data.y)

    # Backward pass e aggiornamento dei pesi
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def new_main(args, model: torch.nn.Module, datasets: List[Data], loss_fn, optimizer: torch.optim.Optimizer, config_folder: str, 
         train_datasets: List[Data], val_datasets: List[Data]=None, test_datasets: List[Data]=None):
    
    train_loss_list, dev_loss_list, eval_accuracy_list = [], [], []
    best_accuracy = 0.0
    scaler = torch.cuda.amp.GradScaler('cuda', enabled=args.use_fp16)

    # Lista di epoche specificate dall'utente (ad esempio, [10, 20, 30])
    epoch_list = args.epochs  # Ora `args.epochs` è una lista di epoche.
    total_epochs = 0  # Tiene traccia del numero totale di epoche eseguite finora

    # Itera su ciascun valore della lista di epoche
    for max_epochs in epoch_list:
        # Ciclo di addestramento fino a `max_epochs`
        for epoch in tqdm(range(total_epochs, max_epochs), desc=f"Training until epoch {max_epochs} ..."):
            train_loss: float = train(model, loss_fn, optimizer, scaler, *train_datasets)
            train_loss_list.append(train_loss)

            accuracy, dev_loss, dev_pred = evaluate(model, "dev", *datasets)
            dev_loss = dev_loss.item()
            dev_loss_list.append(dev_loss)
            eval_accuracy_list.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = f"{config_folder}/{args.dataset_name}_epochs-{max_epochs}.pth"
                torch.save(model.state_dict(), best_model_path)
                config = read_json(f"{config_folder}/config.json")
                config["checkpoint_accuracy"] = best_accuracy
                config["checkpoint_epoch"] = epoch + 1  # Epoch count starts at 1
                config["checkpoint_loss"] = dev_loss

                write_json(f"{config_folder}/config.json", config)

        # Aggiorna il numero totale di epoche eseguite
        total_epochs = max_epochs

        # Valutazione finale per l'epoca corrente (`max_epochs`)
        model.load_state_dict(torch.load(best_model_path))
        test_accuracy, test_loss, test_pred = evaluate(model, "test", *datasets)

        # Salva il modello e i risultati della fase attuale
        config = read_json(f"{config_folder}/config.json")
        config["test_accuracy"] = test_accuracy
        config["test_loss"] = test_loss.item()
        config["test_pred"] = test_pred.tolist()
        config["train_loss_list"] = train_loss_list
        config["dev_loss_list"] = dev_loss_list
        config["dev_accuracy_list"] = eval_accuracy_list

        # Scrive i risultati in un file JSON specifico per il numero di epoche
        output_config_folder = f"{config_folder}/epochs-{max_epochs}"
        os.makedirs(output_config_folder, exist_ok=True)
        write_json(f"{output_config_folder}/config.json", config)

    print(f"Training complete for all epochs: {epoch_list}")


def main(args, model: torch.nn.Module, datasets: List[Data], loss_fn, optimizer: torch.optim.Optimizer, config_folder: str, 
         train_datasets: List[Data], val_datasets: List[Data]=None, test_datasets: List[Data]=None):
    
    train_loss_list, dev_loss_list, eval_accuracy_list = [], [], []
    best_accuracy = 0.0
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_fp16)

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Training ..."):
        train_loss: float = train(model, loss_fn, optimizer, scaler, *train_dataset)
        train_loss_list.append(train_loss)
        
        accuracy, dev_loss, dev_pred = evaluate(model, "dev", *datasets)
        dev_loss = dev_loss.item()
        dev_loss_list.append(dev_loss)
        eval_accuracy_list.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"{config_folder}/{args.dataset_name}.pth")
            config = read_json(f"{config_folder}/config.json")
            config["checkpoint_accuracy"] = best_accuracy
            config["checkpoint_epoch"] = epoch
            config["checkpoint_loss"] = dev_loss

            write_json(f"{config_folder}/config.json", config)

    # Evaluate on test datasets using model with best accuracy on dev set
    model.load_state_dict(torch.load(f"{config_folder}/{args.dataset_name}.pth"))
    test_accuracy, test_loss, test_pred = evaluate(model, "test", *datasets)
    
    config = read_json(f"{config_folder}/config.json")
    config["test_accuracy"] = test_accuracy
    config["test_loss"] = test_loss.item()
    config["test_pred"] = test_pred.tolist()
    config["train_loss_list"] = train_loss_list
    config["dev_loss_list"] = dev_loss_list
    config["dev_accuracy_list"] = eval_accuracy_list
    write_json(f"{config_folder}/config.json", config)

if __name__ == "__main__":
    # Parsing command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, nargs='+', default=[10, 13, 19, 26, 37, 51, 71, 100, 138, 193, 268, 372, 517, 719, 1000])
    parser.add_argument('--t', type=float, nargs='+', default=[0.0, 0.0])  # Lista di t
    parser.add_argument('--dim_reduction', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset_name', type=str,  default='citeseer')#required=True,
    parser.add_argument('--gnn_type', type=str,  default='GAT')#required=True,
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--root_project', type=str, default="")
    parser.add_argument('--use_fp16', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    seed_everything(args.seed)
    config_folder: str = dump_configuration_new(vars(args))

    USE_SUBGRAPH = True

    dimension_dataset: dict = {"cora": {"dim0": 1433, "out_dim": 7}, "citeseer": {"dim0": 3703, "out_dim": 6}, "pubmed": {"dim0": 500, "out_dim": 3}, "Computers":{"dim0": 767, "out_dim": 10}, "Photo":{"dim0": 745, "out_dim": 8}, "Europe":{"dim0":399, "out_dim":4}}
    path_to_folder: str = f"/home/User/projects/RG_GNN/RGNN_data4"

    datasets = []
    train_datasets = []
    heat_flag = 0
    for t in args.t:
        dataset = CustomGraphDataset(root=path_to_folder, device=args.device, dataset_name=args.dataset_name, t=t)[0]
        
        if hasattr(dataset, "train_mask") and USE_SUBGRAPH:
            mask = dataset.train_mask[dataset.edge_index[0]] & dataset.train_mask[dataset.edge_index[1]]
            edge_index_subgraph = dataset.edge_index[:, mask]
            train_dataset = Data(x=dataset.x[dataset.train_mask], edge_index=edge_index_subgraph, y=dataset.y[dataset.train_mask])
            train_datasets.append(train_dataset)
        else:
            train_datasets.append(dataset)
        
        datasets.append(dataset)

    dim0, out_dim = dimension_dataset[args.dataset_name]["dim0"], dimension_dataset[args.dataset_name]["out_dim"]
    
    h_dim = dim0
    # args.dim_reduction = False
    args.dim_reduction = True if args.gnn_type == "GAT" else args.dim_reduction
    h_dim_mapping = {}

    if args.dim_reduction and USE_SUBGRAPH:
        h_dim_mapping = {"Computers": 24, "Photo": 64, "pubmed": 128, "citeseer": 1024, "cora": 1024, "Europe":64}
        if args.gnn_type == "GAT":
            h_dim_mapping = {"Computers": 16, "Photo": 64, "pubmed": 32, "citeseer": 1024, "cora": 1024, "Europe":64}
            if args.t == 4:
                h_dim_mapping = {"Computers": 16, "Photo": 64, "pubmed": 16, "citeseer": 1024, "cora": 1024}
    
    h_dim = h_dim_mapping.get(args.dataset_name, dim0)
    print("hdim", h_dim)

    model = MultiGraphModel(
        gnn_type=GNN_TYPE[args.gnn_type],
        num_layers=args.num_layers,
        input_dim=dim0,
        h_dim=h_dim,
        num_classes=out_dim,
        dim_reduction=args.dim_reduction,
        num_graphs=len(args.t)  # Numero di grafi basato sulla lunghezza di t
    )
    model.to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    new_main(args, model, datasets, loss_fn, optimizer, config_folder, train_datasets)
