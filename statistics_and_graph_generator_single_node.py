import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap  # Non utilizzato
import matplotlib.cm as cm
import torch
from graph_dataset import CustomGraphDataset  
from scipy import stats

# Lista dei dataset
# dataset_names = ['cora', 'citeseer', 'pubmed', 'Photo', 'Computers', 'Europe'] # 'Computers'
dataset_names = ['cora']
gnn_type = 'GAT'
lr = 0.0001
num_layers = 2
seed_list_file = 'RG_GNN/10_numbers.txt'
epochs_list_file = 'RG_GNN/epochs_logarithmic.txt'
tau_type_files = []

def perform_statistical_tests(our_scores, baseline_scores, dataset_name, epochs, stat_file):
    if len(our_scores) > 0 and len(baseline_scores) > 0:
        # Controlla la normalità per 'ours' e 'baseline'
        is_ours_normal = check_normality(our_scores, 'ours')
        is_baseline_normal = check_normality(baseline_scores, 'baseline')

        # Se entrambi sono normali, esegui il t-test
        is_ours_normal = False
        is_baseline_normal = False
        if is_ours_normal and is_baseline_normal:
            t_statistic, p_value, is_significant = t_student_test(our_scores, baseline_scores)
            stat_file.write(f"{dataset_name}\t{epochs}\tours\t{is_ours_normal}\t{t_statistic:.4f}\t{p_value:.4f}\t{is_significant}\tT_test\n")
            stat_file.write(f"{dataset_name}\t{epochs}\tbaseline\t{is_baseline_normal}\t{t_statistic:.4f}\t{p_value:.4f}\t{is_significant}\tT_test\n")
        # Altrimenti, esegui il test di Mann-Whitney U
        else:
            u_statistic, p_value, is_significant = wilcoxon_test(our_scores, baseline_scores)
            stat_file.write(f"{dataset_name}\t{epochs}\tours\t{is_ours_normal}\t{u_statistic:.4f}\t{p_value:.4f}\t{is_significant}\tWilcoxon_test\n")
            stat_file.write(f"{dataset_name}\t{epochs}\tbaseline\t{is_baseline_normal}\t{u_statistic:.4f}\t{p_value:.4f}\t{is_significant}\tWilcoxon_test\n")
    else:
        # In caso di punteggi vuoti
        print(f"Attenzione: punteggi mancanti per il dataset {dataset_name} all'epoca {epochs}.")
        stat_file.write(f"{dataset_name}\t{epochs}\tours\tNaN\tNaN\tNaN\tFalse\n")
        stat_file.write(f"{dataset_name}\t{epochs}\tbaseline\tNaN\tNaN\tNaN\tFalse\n")


# Funzione per il test di normalità
def check_normality(scores, model_name):
    stat, p_value = stats.shapiro(scores)
    alpha = 0.05  # livello di significatività
    is_normal = p_value >= alpha  # True se i dati seguono una distribuzione normale
    if is_normal:
        print(f"Non si rifiuta l'ipotesi nulla: i punteggi di '{model_name}' seguono una distribuzione normale (p-value: {p_value:.4f}).")
    else:
        print(f"Si rifiuta l'ipotesi nulla: i punteggi di '{model_name}' non seguono una distribuzione normale (p-value: {p_value:.4f}).")
    print(f"Statistiche del test di Shapiro-Wilk per '{model_name}': {stat:.4f}")
    
    return is_normal  # Ritorna True o False

def t_student_test(ours_scores, baseline_scores):
    t_statistic, p_value = stats.ttest_ind(ours_scores, baseline_scores, alternative='greater')
    alpha = 0.05  # livello di significatività
    is_significant = p_value < alpha  # True se il risultato è significativo
    return t_statistic, p_value, is_significant

def wilcoxon_test(ours_scores, baseline_scores):
    # u_statistic, p_value = stats.mannwhitneyu(ours_scores, baseline_scores, alternative='greater')
    u_statistic, p_value = stats.wilcoxon([float(b) for b in ours_scores], [float(b) for b in baseline_scores], alternative='greater')    
    alpha = 0.05  # livello di significatività
    is_significant = p_value < alpha  # True se il risultato è significativo
    return u_statistic, p_value, is_significant

# Funzione per fare la lista a partire da un file con i valori in una sola colonna
def make_list(input_file):
    output_list = []
    with open(input_file, 'r') as f:
        for line in f:
            val = line.strip()
            output_list.append(int(val))  # Assicurati di convertire a int
    return output_list

# Funzione per calcolare media e deviazione standard
def calculate_mean_variance(prediction_scores):
    mean_score = np.mean(prediction_scores)
    variance_score = np.var(prediction_scores)
    return mean_score, variance_score

# Funzione per ottenere characteristic_tau
def get_characteristic_tau(dataset_name):
    characteristic_tau_file = f"RG_GNN/RGNN_data4/{dataset_name}/{dataset_name}_data_0_tau_type.txt"
    with open(characteristic_tau_file, 'r') as f:
        next(f)
        for line in f:
            tau, tau_type = line.strip().split('\t')
            if tau_type == "maximum":
                return float(tau)
    return None

# Funzione per leggere più file tau_type e ottenere tutte le tuple [0.0, tau] con tau_type == "maximum"
def get_tau_values_from_files(tau_type_files):
    tau_values = []
    for tau_type_file in tau_type_files:
        with open(tau_type_file, 'r') as f:
            next(f)  # Ignora l'header
            for line in f:
                tau, tau_type = line.strip().split('\t')
                if tau_type == "maximum":
                    tau_values.append(([0.0, float(tau)], f"random_{tau_type_file}"))
    return tau_values

# Funzione per calcolare intervallo di confidenza
def calculate_confidence_interval(data, size, confidence=0.68):
    bootstrapped_means = []
    for _ in range(1000):  # Numero di campioni bootstrap
        sample = np.random.choice(data, size=size, replace=True) # size=len(data)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    return lower_bound, upper_bound, bootstrapped_means

# File di output finale (uno solo)
output_file = f'gnn_type-{gnn_type}_lr-{lr}_num_layers-{num_layers}_combined_results_prediction_scores_size_random.txt'
statistical_output_file = f'statistical_results_{gnn_type}_random.txt'  # Nuovo file per i risultati statistici

# Creiamo la lista dei semi e delle epoche
seed_list = make_list(seed_list_file)
epochs_list = make_list(epochs_list_file)

# Apriamo il file di output in modalità scrittura
with open(output_file, 'w') as out_file, open(statistical_output_file, 'w') as stat_file:
    out_file.write("dataset\ttau\tepochs\tgnn_type\tlr\tnum_layers\tmean_score\tstd_score\n")
    stat_file.write("dataset\tepochs\tmodel\tis_normal\tstatistic\tp_value\tsignificant\n")

    # Iteriamo su ogni dataset
    for dataset_name in dataset_names:

        results_for_plot = {}

        characteristic_tau = get_characteristic_tau(dataset_name)
        # characteristic_tau = 4.0
        if characteristic_tau is None:
            print(f"Warning: characteristic_tau not found for dataset {dataset_name}")
            continue
        
        tau_values = [
            ([0.0, 0.0], 'baseline'),  
            ([0.0, characteristic_tau], 'ours')
        ]

        # Aggiungiamo tutte le tuple dai vari file tau_type_file
        tau_values.extend(get_tau_values_from_files(tau_type_files))

        ours_stat_scores = {}
        baseline_stat_scores = {}

        # Iteriamo su ciascuna combinazione di [0.0, tau]
        for tau_max_list, label in tau_values:
            prediction_scores_by_epoch = {}
            test_accuracies_by_epoch = {}
            tau_str = "_".join([str(x) for x in tau_max_list])

            for epochs in epochs_list:
                for seed in seed_list:
                    file_name = f"experiments/{dataset_name}/t-{tau_str}_gnntype-{gnn_type}_lr-{lr}_num_layers-{num_layers}_seed-{seed}_multi_epochs_no_reduction/epochs-{epochs}/config.json"
                    file_name2 = f"experiments/{dataset_name}/t-{tau_str}_gnntype-{gnn_type}_lr-{lr}_num_layers-{num_layers}_seed-{seed}_multi_epochs/epochs-{epochs}/config.json"

                    size_est = []

                    if os.path.exists(file_name):
                        with open(file_name, 'r') as f:
                            data = json.load(f)
                            predictions = data.get('test_pred', None)
                            test_accuracy = data.get('test_accuracy', None)
                            if test_accuracy is not None:
                                if epochs not in test_accuracies_by_epoch:
                                    test_accuracies_by_epoch[epochs] = []
                                test_accuracies_by_epoch[epochs].append(test_accuracy)
                            if predictions is not None:
                                torch.manual_seed(seed)
                                dataset = CustomGraphDataset(root=f"/home/User/projects/RG_GNN/RGNN_data4", device="cpu", dataset_name=dataset_name, t=tau_max_list[1])[0]
                                mask = dataset.test_mask
                                size_est.append(mask.sum())
                                predictions = torch.tensor(predictions).to(device="cpu")
                                prev_score = predictions[mask] == dataset.y[mask]
                                if epochs not in prediction_scores_by_epoch:
                                    prediction_scores_by_epoch[epochs] = []
                                prediction_scores_by_epoch[epochs].extend(prev_score.numpy().tolist())
                    elif os.path.exists(file_name2):
                        with open(file_name2, 'r') as f:
                            data = json.load(f)
                            predictions = data.get('test_pred', None)
                            test_accuracy = data.get('test_accuracy', None)
                            if test_accuracy is not None:
                                if epochs not in test_accuracies_by_epoch:
                                    test_accuracies_by_epoch[epochs] = []
                                test_accuracies_by_epoch[epochs].append(test_accuracy)
                            if predictions is not None:
                                torch.manual_seed(seed)
                                dataset = CustomGraphDataset(root=f"/home/User/projects/RG_GNN/RGNN_data4", device="cpu", dataset_name=dataset_name, t=tau_max_list[1])[0]
                                mask = dataset.test_mask
                                size_est.append(mask.sum())
                                predictions = torch.tensor(predictions).to(device="cpu")
                                prev_score = predictions[mask] == dataset.y[mask]
                                if epochs not in prediction_scores_by_epoch:
                                    prediction_scores_by_epoch[epochs] = []
                                prediction_scores_by_epoch[epochs].extend(prev_score.numpy().tolist())
                    else:
                        print(f"File not found: {file_name}")

            # Calcola media, varianza e intervallo di confidenza per ogni epoca
            for epochs, scores in prediction_scores_by_epoch.items():
                if scores:  # Assicurati che ci siano punteggi
                    mean_score, variance_score = calculate_mean_variance(scores)# check 
                    lower_bound, upper_bound, score_for_stat = calculate_confidence_interval(scores, int(np.mean(size_est)))
                    

                    mean_score, variance_score = calculate_mean_variance(test_accuracies_by_epoch[epochs])
                    lower_bound = mean_score - np.sqrt(variance_score)
                    upper_bound = mean_score + np.sqrt(variance_score)

                    # print(np.sqrt(calculate_mean_variance(score_for_stat)))
                    mean_score = np.mean(score_for_stat)

                    if tau_max_list == [0.0, characteristic_tau]:
                        ours_stat_scores[epochs] = scores#score_for_stat #score
                    elif tau_max_list == [0.0, 0.0]:
                        baseline_stat_scores[epochs] = scores#score_for_stat #score

                    out_file.write(f"{dataset_name}\t{tau_max_list}\t{epochs}\t{gnn_type}\t{lr}\t{num_layers}\t{mean_score:.8f}\t{np.sqrt(variance_score):.8f}\n")
                    results_for_plot.setdefault(tuple(tau_max_list), {}).setdefault('scores', []).append({'epochs': epochs, 'mean_score': mean_score, 'lower_bound': lower_bound, 'upper_bound': upper_bound, 'label': label})
        
        for epochs in epochs_list:        
            perform_statistical_tests(ours_stat_scores[epochs], baseline_stat_scores[epochs], dataset_name, epochs, stat_file)

        # Creazione di un grafico per questo dataset
        plt.figure(figsize=(10, 6))
        for tau, results in results_for_plot.items():
            x_values = [result['epochs'] for result in results['scores']]
            y_values = [result['mean_score'] for result in results['scores']]
            lower_bounds = [result['lower_bound'] for result in results['scores']]
            upper_bounds = [result['upper_bound'] for result in results['scores']]
            labels = [result['label'] for result in results['scores']][0]
            
            plt.plot(x_values, y_values, label=labels, marker='o', linestyle='-')


            # Aggiungere intervalli di confidenza come area ombreggiata
            plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2)

        plt.title(f'Accuracy vs Epochs for {dataset_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(f'{dataset_name}_scores_vs_epochs_size_{gnn_type}_random.png')
        plt.close()

print(f"Risultati salvati su {output_file} e grafici generati per ogni dataset.")
