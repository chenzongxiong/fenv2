#!/bin/bash

#SBATCH -J run-lstm-pavel-dima-sequence
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-lstm-dima-sequence-units-256-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-1

# __units__array=(1 8 16 32 64 128 256)
method_array=("debug-pavel" "debug-dima")
function run_pavel {
    __unit__=$1
    host_name=`hostname`
    echo "Run pavel with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
    python lstm/lstm_hysteretical.py --epochs 5000 --force_train --lr 0.001 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ ${__unit__} --method debug-pavel --loss mse --diff-weights
    exit 0
}


function run_dima {
    __unit__=$1
    host_name = `hostname`
    echo "Run dima with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
    python lstm/lstm_hysteretical.py --epochs 5000 --force_train --lr 0.001 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ ${__unit__} --method debug-dima --loss mse --diff-weights
    exit 0
}
function run_256 {
    method=$1
    host_name = `hostname`
    echo "Run ${method} with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
    python lstm/lstm_hysteretical.py --epochs 5000 --force_train --lr 0.01 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ 256 --method ${method} --loss mse --diff-weights
}

# run_pavel ${__units__array[SLURM_ARRAY_TASK_ID]}
# run_dima ${__units__array[SLURM_ARRAY_TASK_ID]}
run_256 ${method_array[SLURM_ARRAY_TASK_ID]}
