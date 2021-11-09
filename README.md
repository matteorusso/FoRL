## Adapted from https://github.com/qqadssp/Pytorch-Large-Scale-Curiosity

## Requirement

Python 3.6+
Pythorch 0.4

## Usage

Download this repo and run run.py

    python run.py --feat_learning none_mlp --env_kind frozenlake --use_oh 1 --env FrozenLake-v0 --ext_coeff 1.0 --int_coeff 0.0 --lr 5e-4 --ent_coeff 1e-2 --envs_per_process 32 --nlumps 2 --norm_rew 0
