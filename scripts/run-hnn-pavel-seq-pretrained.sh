#!/bin/bash

#SBATCH -J run-hnn-dima-sequence-multiple-times
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-dima-sequence-multiple-times-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=40-59

__nb_plays__=(10 25 25 50 50 100)
__units__=(10 10 25 25 50 50)

__nb_plays__array=()
__units__array=()
ensemble_array=()
for j in {0..6}
do
    for i in {1..20}
    do
        __nb_plays__array+=(${__nb_plays__[j]})
        __units__array+=(${__units__[j]})
        ensemble_array+=($i)
    done
done
echo ${__nb_plays__array[*]}
echo ${__units__array[*]}
echo ${ensemble_array[*]}

function run_pavel {
    __nb_plays__=$1
    __units__=$2
    ensemble=$3
    __state__=$4
    host_name=`hostname`
    echo "Run pavel with __nb_plays__=${__nb_plays__}, __units__=${__units__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble ${ensemble}, __state__: ${__state__}"
    source $HOME/.venv3/bin/activate
    python run_hnn_dima_pavel_seq.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-pavel --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --diff-weights --ensemble ${ensemble} --__state__ ${__state__}
}

# function run_dima {
#     __nb_plays__=$1
#     __units__=$2
#     ensemble=$3
#     host_name=`hostname`
#     echo "Run dima with __nb_plays__=${__nb_plays__}, __units__=${__units__}, job id ${SLURM_JOB_ID}, task id ${SLURM_ARRAY_TASK_ID}, hostname ${host_name}, ensemble ${ensemble}"
#     source $HOME/.venv3/bin/activate
#     python run_hnn_dima_pavel_seq.py --epochs 1000 --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-dima --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights --ensemble ${ensemble}
# }

# run_pavel ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} 1
# run_pavel 25 25 1 1

# run_dima ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
for SLURM_ARRAY_TASK_ID in {40..60}
do
    run_pavel ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} 1
done

for SLURM_ARRAY_TASK_ID in {40..60}
do
    run_pavel ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} -1
done

for SLURM_ARRAY_TASK_ID in {40..60}
do
    run_pavel ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} 100
done

for SLURM_ARRAY_TASK_ID in {40..60}
do
    run_pavel ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]} -100
done
