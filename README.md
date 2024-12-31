# On the Expressiveness and Length Generalization of Selective State-Space Models on Regular Languages

### Aleksandar Terzić, Michael Hersche, Giacomo Camposampiero, Thomas Hofmann, Abu Sebastian, Abbas Rahimi

_Official code of the AAAI'25 publication "On the Expressiveness and Length Generalization of Selective State-Space Models on Regular Languages"_


<div align="center">
  <img src='SDSSM_Sketch.jpg' width="50%"/>
</div>


## Requirements

The `conda` software is required for running the code. Generate a new environment with

```
$ conda create --name sdssm_env python=3.10
$ conda activate sdssm_env
```

We used PyTorch 2.5.1 with CUDA. jax and jaxlib are required for the sample generation for the _Arithmetic_ task. 

```
$ (sdssm_env) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
$ (sdssm_env) pip install -r requirements.txt
```

## Experiments

The experiments should be run from the repo directory. Results will be stored in the _results_ directory, unless specified otherwise by using the --experiment_dir flag.

### SD-SSM

For reproducing the _SD-SSM_ results from Table 1, use the following commands:

```
# Parity:
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=parity_check --lr=0.0001 --seed=0 --num_transition_matrices=8 --Lp_norm=1.2 --train_steps=1000000 --state_size=64 --train_length=40

# Even Pairs:
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=even_pairs --lr=0.00002 --seed=0 --num_transition_matrices=8 --Lp_norm=1.4 --train_steps=1000000 --state_size=64 --train_length=40

# Cycle 
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=cycle_navigation --lr=0.00002 --seed=0 --num_transition_matrices=8 --Lp_norm=1.3 --train_steps=1000000 --state_size=64 --train_length=40

# Modular Arithmetic
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=modular_arithmetic --lr=0.0001 --seed=0 --num_transition_matrices=18 --Lp_norm=1.2 --train_steps=1000000 --state_size=64 --train_length=40  

# C2 x C4
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=C2xC4 --lr=0.00002 --seed=0 --num_transition_matrices=6 --Lp_norm=1.3 --train_steps=1000000 --state_size=64 --train_length=40  

# D4
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=D4 --lr=0.0001 --seed=0 --num_transition_matrices=6 --Lp_norm=1.2 --train_steps=1000000 --state_size=64 --train_length=40  

# A5
python experiments/run_experiment.py --architecture=SDSSM --test_length=500 --task=A5 --lr=0.00002 --seed=0 --num_transition_matrices=6 --Lp_norm=1.3 --train_steps=1000000 --state_size=64 --train_length=40  
```

### Complex Diagonal

For reproducing the _Complex Diagonal_ results from Table 1, use the following command with appropriate choice of task and learning rate:

```
# Example on Parity:
python experiments/run_experiment.py --architecture=complex_diagonal_with_B_nonlinear --test_length=500 --task=parity_check --lr=0.0005 --seed=0 --train_steps=1000000 --state_size=64 --train_length=40
```

### Inspection with TensorBoard

For an overview of the training metrics and the length generalization accuracies, the TensorBoard tool can be used. During simulations, data is collected which can be illustrated by the tool in the browser. 

## Acknowledgments

This repository is heavily inspired by the [Neural Networks and the Chomsky Hierarchy repo](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy) and the automaton generating code associated with [Transformers Learn Shortcuts to Automata](https://huggingface.co/datasets/synthseq/automata/tree/main).

## Citation

If you use the work released here for your research, please cite this paper:
```
@inproceedings{terzic2025sdssm,
    Author = {Terzić, Aleksandar and Hersche, Michael and Camposampiero, Giacomo and Hofmann, Thomas and Sebastian, Abu and Rahimi, Abbas },
    Booktitle = {Association for the Advancement of Artificial Intelligence (AAAI)},
    Title = {On the Expressiveness and Length Generalization of Selective State-Space Models on Regular Languages},
    Year = {2025}}
```

## License
Our code is licensed under Apache 2.0. Please refer to the LICENSE file for the licensing of our code. 
