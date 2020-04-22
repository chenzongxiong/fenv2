#!/bin/bash

#SBATCH -J run-paval-dima-sequence
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-pavel-dima-sequence-%a-%j.log
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big,small
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-5

__nb_plays__array=(10 25 25 50 50 100)
__units__array=(10 10 25 25 50 50)


# input=("bill" "ted" "lisa" "42")

# function echoMe {
#     echo "Hello $1 from job $SLURM_JOB_ID, $SLURM_ARRAY_TASK_ID"
#     sleep 1
#     exit 0
# }

# echoMe ${input[SLURM_ARRAY_TASK_ID]}

function run_pavel {
    __nb_plays__=$1
    __units__=$2
    echo "Run pavel with __nb_plays__=${__nb_plays__}, __units__=${__units__}"
    # python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights
    exit 0
}

function run_dima {
    __nb_plays__=$1
    __units__=$2
    host_name=`hostname`
    echo "Run dima with __nb_plays__=${__nb_plays__}, __units__=${__units__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
    source /home/zxchen/.venv3/bin/activate
    python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights
    # sleep 30
    exit 0
}

run_dima ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]}


# hostname

# # # elu pavel sequence
# python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ 10 --__units__ 10 --__activation__ elu --force_train --diff-weights
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
