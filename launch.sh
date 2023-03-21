#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=g100_usr_interactive
#SBATCH --account=uBS23_InfGer
#SBATCH --time=08:00:00
#SBATCH --mem=64G
## SBATCH --ntasks-per-socket=1
#SBATCH --error=ac2.err
#SBATCH --output=a2c.out

python A2C_Grounding.py