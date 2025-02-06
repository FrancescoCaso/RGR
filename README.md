The files mentioned perform the following tasks:

1. **SpecificHeat.py** and **SpecificHeat.ipynb**: These calculate the characteristic scales using spectral entropy values.
2. **DataGeneratorCharacteristic.py**: This preprocesses the graphs by aligning them to the characteristic scales.
3. **new_train.py**: Once the graphs are preprocessed, this script runs the experiments presented in the paper (you can also use **run_cora_all_t_all_epochs** as an example to run multiple epochs on the Cora dataset).
4. **statistics_and_graph_generator_single_node.py**: This script generates metrics, performs statistical tests, and produces the graphs discussed in the paper.

These steps complete the experimental pipeline described in the paper.
