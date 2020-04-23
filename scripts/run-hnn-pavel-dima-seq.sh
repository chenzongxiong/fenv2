#!/bin/bash

#SBATCH -J run-hnn-pavel-dima-sequence-multiple-times
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-dima-sequence-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-120

__nb_plays__=(10 25 25 50 50 100)
__units__=(10 10 25 25 50 50)

__nb_plays__array=()
__units__array=()
ensemble_array=()
for i in {1..20}
do
    for j in {1..6}
    do
        __nb_plays__array+=(${__nb_plays__[j]})
        __units__array+=(${__units__[j]})
        ensemble_array+=($i)
    done
done
echo $__nb_plays__array
echo $__units__array
echo $ensemble_array

function run_pavel {
    __nb_plays__=$1
    __units__=$2
    ensemble=$3
    echo "Run pavel with __nb_plays__=${__nb_plays__}, __units__=${__units__}"
    # python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights --ensemble ${ensemble}
}

function run_dima {
    __nb_plays__=$1
    __units__=$2
    ensemble=$3
    host_name=`hostname`
    echo "Run dima with __nb_plays__=${__nb_plays__}, __units__=${__units__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}"
    source $HOME/.venv3/bin/activate
    # python run_hnn_dima_pavel_seq.py --epochs 1000 --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights --ensemble ${ensemble}
    # exit 0
}

run_dima ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
