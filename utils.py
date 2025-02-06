

import os
import json
import random
import torch

def dump_configuration_new(cfg: dict) -> None:
    config = cfg.copy()
    del config["device"]
    config["gnn_type"] = str(config["gnn_type"])

    # if cfg["dataset_name"] == "synthetic/":
    #     folder_name: str = f'{config["dataset_name"]}/is_ren--{config["is_ren"]}_r--{config["r"]}_t-{config["t"]}_gnntype-{config["gnn_type"]}_lr-{config["lr"]}/'
    # else:
    folder_name: str = f'{config["dataset_name"]}/t-'+"_".join([str(x) for x in config["t"]])+f'_gnntype-{config["gnn_type"]}_lr-{config["lr"]}_num_layers-{config["num_layers"]}_seed-{config["seed"]}_multi_epochs'
    

    config_folder: str = f"{config['root_project']}experiments/{folder_name}"
    os.makedirs(config_folder, exist_ok=True)
    
    write_json(f"{config_folder}/config.json", config)
    return config_folder

def dump_configuration_different_t(cfg: dict) -> None:
    config = cfg.copy()
    del config["device"]
    config["gnn_type"] = str(config["gnn_type"])

    # if cfg["dataset_name"] == "synthetic/":
    #     folder_name: str = f'{config["dataset_name"]}/is_ren--{config["is_ren"]}_r--{config["r"]}_t-{config["t"]}_gnntype-{config["gnn_type"]}_lr-{config["lr"]}/'
    # else:
    folder_name: str = f'{config["dataset_name"]}/'+f'_gnntype-{config["gnn_type"]}_lr-{config["lr"]}_num_layers-{config["num_layers"]}_seed-{config["seed"]}_multi_epochs'
    

    config_folder: str = f"{config['root_project']}experiments/{folder_name}"
    os.makedirs(config_folder, exist_ok=True)
    
    write_json(f"{config_folder}/config.json", config)
    return config_folder

def dump_configuration(cfg: dict) -> None:
    config = cfg.copy()
    del config["device"]
    config["gnn_type"] = str(config["gnn_type"])

    if cfg["dataset_name"] == "synthetic/":
        folder_name: str = f'{config["dataset_name"]}/is_ren--{config["is_ren"]}_r--{config["r"]}_t-{config["t"]}_gnntype-{config["gnn_type"]}_lr-{config["lr"]}/'
    else:
        folder_name: str = f'{config["dataset_name"]}/t-{config["t"]}_gnntype-{config["gnn_type"]}_lr-{config["lr"]}_num_layers-{config["num_layers"]}_seed-{config["seed"]}_epochs-{config["epochs"]}{"_dualgraph" if config["dual_graph"] else ""}/'
    

    config_folder: str = f"{config['root_project']}/experiments/{folder_name}/"
    os.makedirs(config_folder, exist_ok=True)
    
    write_json(f"{config_folder}/config.json", config)
    return config_folder

def write_json(filename: str, dictionary: dict) -> None:
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)

def read_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)

def dump_model(model: torch.nn.Module, config_folder: str, best_accuracy: float, epoch: int, epoch_loss_print: float) -> None:
    torch.save(model.state_dict(), f"{config_folder}/best_model.pt")
    current_config = read_json(f"{config_folder}/config.json")
    current_config["checkpoint_accuracy"] = best_accuracy
    current_config["checkpoint_epoch"] = epoch
    current_config["checkpoint_loss"] = epoch_loss_print
    write_json(f"{config_folder}/config.json", current_config)



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False