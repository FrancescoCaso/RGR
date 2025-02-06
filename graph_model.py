#!/usr/bin/python3

from enum import Enum, auto
import torch 
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, TransformerConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool  # Pooling methods for graph classification

class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    GTR = auto()

    def __str__(self):
        if self is GNN_TYPE.GCN:
            return "GCN"
        elif self is GNN_TYPE.GGNN:
            return "GGNN"
        elif self is GNN_TYPE.GIN:
            return "GIN"
        elif self is GNN_TYPE.GAT:
            return "GAT"
        elif self is GNN_TYPE.GTR:
            return "GTR"
        assert False
        

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.GTR:
            num_heads = 4
            return TransformerConv(in_dim, out_dim // num_heads, heads=num_heads)

class MultiGraphClassificationModel(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction: bool, num_graphs: int):
        super(MultiGraphClassificationModel, self).__init__()
        self.num_graphs = num_graphs  # Numero di grafi in input
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.layer_norms = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.dim_reduction = dim_reduction
        
        if dim_reduction:
            self.reduce_node_features = nn.ModuleList([nn.Linear(input_dim, h_dim) for _ in range(num_graphs)])
        else:
            h_dim = input_dim

        # Creiamo i layer di convoluzione e layer norm per ciascun grafo
        for _ in range(num_layers):
            for i in range(num_graphs):
                self.layers[i].append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
                self.layer_norms[i].append(nn.LayerNorm(h_dim))
        
        # La dimensione del classificatore è ora proporzionale al numero dei grafi
        self.classifier = torch.nn.Linear(h_dim * num_graphs, num_classes)

    def forward(self, *graph_data_list: Data):
        assert len(graph_data_list) == self.num_graphs, "Il numero dei grafi in input deve corrispondere al parametro num_graphs"

        # Riduzione dimensionale (se necessaria) e preparazione delle feature per ogni grafo
        graph_features = []
        for i, data in enumerate(graph_data_list):
            x, edge_index = data.x, data.edge_index
            batch = data.batch  # Necessario per il pooling globale
            
            if self.dim_reduction:
                x = self.reduce_node_features[i](x)
            
            # Passiamo attraverso i layer e le connessioni residuali
            for layer, layer_norm in zip(self.layers[i], self.layer_norms[i]):
                conv_x = layer(x, edge_index)
                conv_x = layer_norm(conv_x)
                conv_x = F.relu(conv_x)
                x = x + conv_x  # Connessione residuale

            # Applica il pooling globale per ottenere la rappresentazione del grafo
            pooled_x = global_mean_pool(x, batch)  # Puoi anche usare global_max_pool per un pooling massimo
            graph_features.append(pooled_x)

        # Concatenazione delle rappresentazioni di tutti i grafi
        x = torch.cat(graph_features, dim=-1)

        # Classificatore finale (classificazione del grafo)
        logits = self.classifier(x)
        return logits

class MultiGraphModelWithNodeAttention(nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction: bool, num_graphs: int):
        super(MultiGraphModelWithNodeAttention, self).__init__()
        self.num_graphs = num_graphs
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.layer_norms = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.dim_reduction = dim_reduction

        if dim_reduction:
            self.reduce_node_features = nn.ModuleList([nn.Linear(input_dim, h_dim) for _ in range(num_graphs)])
        else:
            h_dim = input_dim

        # Creiamo i layer di convoluzione e layer norm per ciascun grafo
        for _ in range(num_layers):
            for i in range(num_graphs):
                self.layers[i].append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
                self.layer_norms[i].append(nn.LayerNorm(h_dim))

        # Meccanismo di attenzione
        self.attention_fc = nn.Linear(h_dim, 1)  # Calcola il punteggio di attenzione per ogni nodo

    def forward(self, *graph_data_list: Data):
        assert len(graph_data_list) == self.num_graphs, "Il numero dei grafi in input deve corrispondere al parametro num_graphs"

        all_graph_features = []
        
        for i, data in enumerate(graph_data_list):
            x, edge_index = data.x, data.edge_index
            if self.dim_reduction:
                x = self.reduce_node_features[i](x)

            # Passiamo attraverso i layer e le connessioni residuali
            for layer, layer_norm in zip(self.layers[i], self.layer_norms[i]):
                conv_x = layer(x, edge_index)
                conv_x = layer_norm(conv_x)
                conv_x = F.relu(conv_x)
                x = x + conv_x  # Connessione residuale

            # Calcoliamo i punteggi di attenzione per ogni nodo
            attention_scores = self.attention_fc(x)  # Shape: (num_nodes, 1)
            attention_weights = F.softmax(attention_scores, dim=0)  # Normalizziamo per ottenere i pesi

            # Media ponderata delle feature dei nodi per ottenere una rappresentazione globale
            weighted_graph_rep = (attention_weights * x).sum(dim=0)  # Shape: (h_dim,)

            all_graph_features.append(weighted_graph_rep)

        # Concatenazione delle rappresentazioni ponderate di tutti i grafi
        combined_features = torch.cat(all_graph_features, dim=0)  # Shape: (num_graphs * h_dim,)

        # Classificatore finale
        logits = self.classifier(combined_features)  # Usa un classificatore per ottenere i logits

        return logits


class MultiGraphModel2(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, num_graphs: int, dim_reduction: bool):
        super(MultiGraphModel, self).__init__()
        
        self.num_graphs = num_graphs
        self.h_dim = h_dim
        self.gnn_type = str(gnn_type)
        self.dim_reduction = dim_reduction
        
        # Liste per layer, normalizzatori e riduttori di dimensione per ciascun grafo
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.layer_norms = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        
        if dim_reduction:
            self.reduce_node_features = nn.ModuleList([nn.Linear(input_dim, h_dim) for _ in range(num_graphs)])
        else:
            h_dim = input_dim  # Se non riduciamo la dimensione, manteniamo la dimensione originale
        
        # Aggiungiamo i layer e le normalizzazioni per ciascun grafo
        for g in range(num_graphs):
            for _ in range(num_layers):
                self.layers[g].append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
                self.layer_norms[g].append(nn.LayerNorm(h_dim))

        # Il classificatore ora prende come input la concatenazione di tutte le rappresentazioni
        self.classifier = torch.nn.Linear(h_dim * num_graphs, num_classes)

    def forward(self, *graph_data):
        """
        :param graph_data: una lista di oggetti Data (uno per ogni grafo) da passare attraverso il modello.
        """
        if len(graph_data) != self.num_graphs:
            raise ValueError(f"Expected {self.num_graphs} graphs, but got {len(graph_data)}.")

        # Liste per memorizzare le rappresentazioni finali di ciascun grafo
        graph_representations = []
        
        # Per ciascun grafo, applichiamo i layer e i normali con connessioni residuali
        for g in range(self.num_graphs):
            x, edge_index = graph_data[g].x, graph_data[g].edge_index

            if self.dim_reduction:
                # Riduzione della dimensionalità delle caratteristiche del nodo
                x = self.reduce_node_features[g](x)

            for layer, layer_norm in zip(self.layers[g], self.layer_norms[g]):
                conv_x = layer(x, edge_index)
                conv_x = layer_norm(conv_x)
                conv_x = F.relu(conv_x)
                x = x + conv_x  # Connessione residuale

            # Aggiungiamo la rappresentazione del grafo alla lista
            graph_representations.append(x)

        # Concatenazione delle rappresentazioni di tutti i grafi
        x_concat = torch.cat(graph_representations, dim=-1)

        # Classificatore finale
        logits = self.classifier(x_concat)
        return logits


class MultiGraphModel(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction: bool, num_graphs: int):
        super(MultiGraphModel, self).__init__()
        self.num_graphs = num_graphs  # Numero di grafi in input
        self.layers = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.layer_norms = nn.ModuleList([nn.ModuleList() for _ in range(num_graphs)])
        self.dim_reduction = dim_reduction
        
        if dim_reduction:
            self.reduce_node_features = nn.ModuleList([nn.Linear(input_dim, h_dim) for _ in range(num_graphs)])
        else:
            h_dim = input_dim

        # Creiamo i layer di convoluzione e layer norm per ciascun grafo
        for _ in range(num_layers):
            for i in range(num_graphs):
                self.layers[i].append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
                self.layer_norms[i].append(nn.LayerNorm(h_dim))
        
        # La dimensione del classificatore è ora proporzionale al numero dei grafi
        self.classifier = torch.nn.Linear(h_dim * num_graphs, num_classes)

    def forward(self, *graph_data_list: Data):
        assert len(graph_data_list) == self.num_graphs, "Il numero dei grafi in input deve corrispondere al parametro num_graphs"

        # Riduzione dimensionale (se necessaria) e preparazione delle feature per ogni grafo
        graph_features = []
        for i, data in enumerate(graph_data_list):
            x, edge_index = data.x, data.edge_index
            if self.dim_reduction:
                x = self.reduce_node_features[i](x)
            
            # Passiamo attraverso i layer e le connessioni residuali
            for layer, layer_norm in zip(self.layers[i], self.layer_norms[i]):
                conv_x = layer(x, edge_index)
                conv_x = layer_norm(conv_x)
                conv_x = F.relu(conv_x)
                x = x + conv_x  # Connessione residuale

            graph_features.append(x)

        # Concatenazione delle rappresentazioni di tutti i grafi
        x = torch.cat(graph_features, dim=-1)

        # Classificatore finale
        logits = self.classifier(x)
        return logits



class TripleGraphModel(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction: bool):
        super(TripleGraphModel, self).__init__()
        self.layers_renormalized = nn.ModuleList()
        self.layers_vanilla = nn.ModuleList()
        self.layers_third = nn.ModuleList()
        
        self.layer_norms_renormalized = nn.ModuleList()
        self.layer_norms_vanilla = nn.ModuleList()
        self.layer_norms_third = nn.ModuleList()
        
        self.h_dim = h_dim
        self.gnn_type = str(gnn_type)
        self.dim_reduction = dim_reduction
        
        if dim_reduction:
            self.reduce_node_features = nn.Linear(input_dim, h_dim)
            self.reduce_node_features_vanilla = nn.Linear(input_dim, h_dim)
            self.reduce_node_features_third = nn.Linear(input_dim, h_dim)
        else:
            h_dim = input_dim

        for _ in range(num_layers):
            # Aggiungiamo il terzo insieme di layer
            self.layers_renormalized.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
            self.layers_vanilla.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
            self.layers_third.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))

            # Aggiungiamo il terzo insieme di normali layer
            self.layer_norms_renormalized.append(nn.LayerNorm(h_dim))
            self.layer_norms_vanilla.append(nn.LayerNorm(h_dim))
            self.layer_norms_third.append(nn.LayerNorm(h_dim))
        
        # La dimensione complessiva del classificatore ora è 3 volte h_dim
        self.classifier = torch.nn.Linear(h_dim * 3, num_classes)

    def forward(self, data_renormalized: Data, data_vanilla: Data, data_third: Data):
        x_renormalized, edge_index_renormalized = data_renormalized.x, data_renormalized.edge_index
        x_vanilla, edge_index_vanilla = data_vanilla.x, data_vanilla.edge_index
        x_third, edge_index_third = data_third.x, data_third.edge_index

        if self.dim_reduction:
            # Riduzione della dimensionalità delle caratteristiche del nodo
            x_renormalized = self.reduce_node_features(x_renormalized)
            x_vanilla = self.reduce_node_features_vanilla(x_vanilla)
            x_third = self.reduce_node_features_third(x_third)

        # Applicazione di ciascun layer ai tre grafi con connessioni residuali
        for layer_renormalized, layer_norm_renormalized, layer_vanilla, layer_norm_vanilla, layer_third, layer_norm_third in zip(self.layers_renormalized, self.layer_norms_renormalized, self.layers_vanilla, self.layer_norms_vanilla, self.layers_third, self.layer_norms_third):
            
            conv_x_renormalized = layer_renormalized(x_renormalized, edge_index_renormalized)
            conv_x_renormalized = layer_norm_renormalized(conv_x_renormalized)
            conv_x_renormalized = F.relu(conv_x_renormalized)
            x_renormalized = x_renormalized + conv_x_renormalized  # Connessione residuale per il grafo renormalizzato

            conv_x_vanilla = layer_vanilla(x_vanilla, edge_index_vanilla)
            conv_x_vanilla = layer_norm_vanilla(conv_x_vanilla)
            conv_x_vanilla = F.relu(conv_x_vanilla)
            x_vanilla = x_vanilla + conv_x_vanilla  # Connessione residuale per il grafo vanilla
            
            conv_x_third = layer_third(x_third, edge_index_third)
            conv_x_third = layer_norm_third(conv_x_third)
            conv_x_third = F.relu(conv_x_third)
            x_third = x_third + conv_x_third  # Connessione residuale per il terzo grafo

        # Concatenazione delle rappresentazioni dei tre grafi
        x = torch.cat((x_renormalized, x_vanilla, x_third), dim=-1)

        # Classificatore finale
        logits = self.classifier(x)
        return logits




class DualGraphModel(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction: bool):
        super(DualGraphModel, self).__init__()
        self.layers_renormalized = nn.ModuleList()
        self.layers_vanilla = nn.ModuleList()
        self.layer_norms_renormalized = nn.ModuleList()
        self.layer_norms_vanilla = nn.ModuleList()
        self.h_dim = h_dim
        self.gnn_type = str(gnn_type)
        self.dim_reduction = dim_reduction
        if dim_reduction:
            self.reduce_node_features = nn.Linear(input_dim, h_dim)
            self.reduce_node_features_vanilla = nn.Linear(input_dim, h_dim)
        else:
            h_dim = input_dim



        for _ in range(num_layers):
            self.layers_renormalized.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
            self.layers_vanilla.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))

            self.layer_norms_renormalized.append(nn.LayerNorm(h_dim))
            self.layer_norms_vanilla.append(nn.LayerNorm(h_dim))
        self.classifier = torch.nn.Linear((h_dim) * 2, num_classes)

    def forward(self, data_renormalized: Data, data_vanilla: Data):
        x_renormalized, edge_index_renormalized = data_renormalized.x, data_renormalized.edge_index
        x_vanilla, edge_index_vanilla = data_vanilla.x, data_vanilla.edge_index

        if self.dim_reduction:
            # Reduce the dimensionality of the node features
            x_renormalized = self.reduce_node_features(x_renormalized)
            x_vanilla = self.reduce_node_features_vanilla(x_vanilla)
            #x_renormalized = F.relu(x_renormalized)
            #x_vanilla = F.relu(x_vanilla)

        for layer_renormalized, layer_norm_renormalized, layer_vanilla, layer_norm_vanilla in zip(self.layers_renormalized, self.layer_norms_renormalized, self.layers_vanilla, self.layer_norms_vanilla):
            conv_x_renormalized = layer_renormalized(x_renormalized, edge_index_renormalized)
            conv_x_renormalized = layer_norm_renormalized(conv_x_renormalized)
            conv_x_renormalized = F.relu(conv_x_renormalized)
            x_renormalized = x_renormalized + conv_x_renormalized  # Residual connection for renormalized graph

            conv_x_vanilla = layer_vanilla(x_vanilla, edge_index_vanilla)
            conv_x_vanilla = layer_norm_vanilla(conv_x_vanilla)
            conv_x_vanilla = F.relu(conv_x_vanilla)
            x_vanilla = x_vanilla + conv_x_vanilla  # Residual connection for vanilla graph

        # Concatenating the representations
        x = torch.cat((x_renormalized, x_vanilla), dim=-1)

        logits = self.classifier(x)
        return logits




class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type: GNN_TYPE, num_layers: int, input_dim: int, h_dim: int, num_classes: int, dim_reduction:bool):
        super(GraphModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dim_reduction: bool = dim_reduction
        if dim_reduction:
            self.reduce_node_features = nn.Linear(input_dim, h_dim)# da rivedere la riduzionelineare
        else:
            h_dim = input_dim

        for _ in range(num_layers):
            self.layers.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))
            self.layer_norms.append(nn.LayerNorm(h_dim))
        self.classifier = torch.nn.Linear(h_dim, num_classes)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index 

        if self.dim_reduction:
            # Reduce the dimensionality of the node features
            x = self.reduce_node_features(x)
            #x = F.relu(x)

        for layer, layer_norm in zip(self.layers, self.layer_norms):
            conv_x = layer(x, edge_index)
            conv_x = layer_norm(conv_x)
            conv_x = F.relu(conv_x)
            x = x + conv_x  # Residual connection

        logits = self.classifier(x)
        return logits





