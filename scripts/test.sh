#!/bin/sh
#
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=8GB

#SBATCH -o fake_%j.log
#SBATCH -e fake_%j.err
#SBATCH -J ageas_test
#SBATCH -t 00:30:00

# echo "#########"
# echo "Print the current environment (verbose)"
# env

echo "#########"
echo "Show information on nvidia device(s)"
nvidia-smi

echo "#########"
echo "Start Testing"

python3 test.py 
echo "#########"