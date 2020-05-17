#!/bin/bash

#SBATCH -J run-lstm-stock-sequence
#SBATCH -D /data/scratch/zxchen/fenv2
#SBATCH -o ./tmp/run-lstm-tanh-stock-sequence-multiple-times-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-8

__units__=(64 128 256)
sigma=(10 20 110)
__units__array=()
ensemble_array=()
sigma_array=()
for i in {1..20}
do
    for k in {0..2}
    do
        for j in {0..2}
        do
            __units__array+=(${__units__[j]})
            ensemble_array+=($i)
            sigma_array+=(${sigma[k]})
        done
    done
done
echo ${__units__array[*]}
echo ${ensemble_array[*]}
echo ${sigma_array[*]}
# method_array=("debug-pavel" "debug-dima")

function run {
    __unit__=$1
    ensemble=$2
    sigma=$3
    host_name=`hostname`
    echo "Run pavel with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble=${ensemble}, sigma={$sigma}"
    source /home/zxchen/.venv3/bin/activate
    python lstm/lstm_hysteretical.py --epochs 30000 --force_train --lr 0.001 --mu 0 --sigma ${sigma} --units 0 --nb_plays 0 --points 0 --__units__ ${__unit__} --method stock --loss mle --diff-weights --ensemble ${ensemble} --__sigma__ ${sigma}
}

# function run_dima {
#     __unit__=$1
#     ensemble=$2
#     host_name=`hostname`
#     echo "Run dima with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble ${ensemble}"
#     source /home/zxchen/.venv3/bin/activate
#     python lstm/lstm_hysteretical.py --epochs 6000 --force_train --lr 0.01 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ ${__unit__} --method debug-dima --loss mse --diff-weights --ensemble ${ensemble}
# }

# function run_256 {
#     method=$1
#     host_name=`hostname`
#     echo "Run ${method} with  __units__=${__unit__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
#     python lstm/lstm_hysteretical.py --epochs 5000 --force_train --lr 0.01 --mu 0 --sigma 0 --units 50 --nb_plays 50 --points 1000 --__units__ 256 --method ${method} --loss mse --diff-weights
# }

# run_pavel ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
# run_dima ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
# run_256 ${method_array[SLURM_ARRAY_TASK_ID]}

run ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} ${sigma_array[SLURM_ARRAY_TASK_ID]}
