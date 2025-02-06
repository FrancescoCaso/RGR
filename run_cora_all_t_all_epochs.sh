#!/usr/bin/env bash

dataset="cora"
gnn_type="GAT"
lr=0.0001
num_layers=2
device="cuda:1"
tau_type_file='/home/User/projects/RG_GNN/RGNN_data4/cora/cora_data_0_tau_type.txt'
seed_file='/home/User/projects/RG_GNN/10_numbers.txt'
epochs_file='/home/User/projects/RG_GNN/epochs_logarithmic.txt'

# Inizializza le liste
tau_list=(0.0)
tau_ctrl=(0.0)

# Legge il file tau_type.txt, ignorando la prima riga (header)
while IFS=$'\t' read -r tau type; do

    if [ "${type}" == "maximum" ]; then
        tau_list+=("${tau}")  # Aggiunge il valore di tau alla lista
        tau_ctrl+=(0.0)       # Aggiunge 0.0 per la lista di controllo
        break
    fi
done < <(tail -n +2 "${tau_type_file}")

# Debug: Stampa la lista finale di tau e tau_ctrl
echo "Final tau_list: ${tau_list[@]}"
echo "Final tau_ctrl: ${tau_ctrl[@]}"

# Converte le liste in stringhe separate da virgole
tau_list_str=$(IFS=, ; echo "${tau_list[*]}")  # Converte la lista in una stringa separata da virgole
tau_ctrl_str=$(IFS=, ; echo "${tau_ctrl[*]}") 

# Inizializza una lista di epochs
epochs_list=()
while IFS= read -r epochs; do
    epochs_list+=("${epochs}")
done < "${epochs_file}"

# Converte la lista di epoche in una stringa separata da virgole
epochs_list_str=$(IFS=, ; echo "${epochs_list[*]}")

# Legge il file dei seed
while IFS= read -r seed; do
# seed=0
    echo "With seed=${seed}"
    # echo "ok"

    # Se la lista di tau non è vuota, avvia l'addestramento
    if [ ${#tau_list[@]} -ne 0 ]; then
        echo "Training with tau_list=${tau_list_str}, seed=${seed}, epochs_list=${epochs_list_str}"

        # echo "okok"
        # Training in multi-mode
        echo "multi-mode"
        python3 RG_GNN/new_train.py --dataset_name ${dataset} --device ${device} --gnn_type ${gnn_type} --t ${tau_list[@]} --lr ${lr} --epochs ${epochs_list[@]} --num_layers ${num_layers} --seed ${seed}

        # Training in control mode
        echo "control"
        python3 RG_GNN/new_train.py --dataset_name ${dataset} --device ${device} --gnn_type ${gnn_type} --t ${tau_ctrl[@]} --lr ${lr} --epochs ${epochs_list[@]} --num_layers ${num_layers} --seed ${seed}
    fi

done < "${seed_file}"

exit 0
