#!/bin/bash

#SBATCH -J run-lstm-dima-sequence
#SBATCH -D /data/scratch/zxchen/fenv2
#SBATCH -o ./tmp/run-lstm-dima-sequence-multiple-times-epochs-2500-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-119

__units__=(1 8 16 32 64 128)
__units__array=()
ensemble_array=()
for j in {0..5}
do
    for i in {1..20}
    do
        __units__array+=(${__units__[j]})
        ensemble_array+=($i)
    done
done
echo ${__units__array[*]}
echo ${ensemble_array[*]}

method_array=("debug-pavel" "debug-dima")

function run_pavel {
    __unit__=$1
    ensemble=$2
    host_name=`hostname`
    echo "Run pavel with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble=${ensemble}"
    source /home/zxchen/.venv3/bin/activate
    python lstm/lstm_hysteretical.py --epochs 3000 --force_train --lr 0.001 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ ${__unit__} --method debug-pavel --loss mse --diff-weights --ensemble ${ensemble}
}

function run_dima {
    __unit__=$1
    ensemble=$2
    host_name=`hostname`
    echo "Run dima with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble ${ensemble}"
    source /home/zxchen/.venv3/bin/activate
    python lstm/lstm_hysteretical.py --epochs 2500 --force_train --lr 0.01 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ ${__unit__} --method debug-dima --loss mse --diff-weights --ensemble ${ensemble}
}

# function run_256 {
#     method=$1
#     host_name=`hostname`
#     echo "Run ${method} with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
#     python lstm/lstm_hysteretical.py --epochs 5000 --force_train --lr 0.01 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ 256 --method ${method} --loss mse --diff-weights
# }

# run_pavel ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
run_dima ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
# run_256 ${method_array[SLURM_ARRAY_TASK_ID]}
