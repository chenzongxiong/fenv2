#!/bin/bash

#SBATCH -J run-paval-dima-sequence
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run_pavel-dima-sequence.log
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=30-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com


hostname
source /home/zxchen/.venv3/bin/activate

# # elu pavel sequence
python run_hnn_dima_pavel_seq.py --epochs 1000 --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 10 --__units__ 10 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 25 --__units__ 10 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 25 --__units__ 25 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 50 --__units__ 25 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 50 --__units__ 50 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 100 --__units__ 50 --__activation__ elu --force_train --diff-weights

# # # elu dima sequence
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 10 --__units__ 10 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 25 --__units__ 10 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 25 --__units__ 25 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 50 --__units__ 25 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 50 --__units__ 50 --__activation__ elu --force_train --diff-weights
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ 100 --__units__ 50 --__activation__ elu --force_train --diff-weights
