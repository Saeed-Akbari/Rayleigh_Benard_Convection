#!/bin/bash
#SBATCH -p batch
#SBATCH -t 60:00:00
#SBATCH --job-name="rbc_2048_1024_1.0e+9"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --mail-user=supawar@okstate.edu
##SBATCH --mail-type=end

module load cuda
module load anaconda3/2020.07
source activate python36
cd /scratch/suraj/RBC/NS2D_WS
python ns2d_ws_rbc_argparse.py config/rbc_parameters_2048_1024_1.0e+9.yaml > logs/rbc_2048_1024_1.0e+9.log
