#!/bin/bash
#SBATCH -p batch
#SBATCH -t 48:00:00
#SBATCH --job-name="rbc_1024_512_1.0e+10"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load cuda
module load anaconda3/2020.07
source activate python36
cd /scratch/suraj/RBC/NS2D_WS
python ns2d_ws_rbc_argparse.py config/rbc_parameters_1024_512_1.0e+10.yaml > logs/rbc_1024_512_1.0e+10.log
