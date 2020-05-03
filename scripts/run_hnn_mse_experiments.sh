#!/bin/bash

#SBATCH -J run-hnn-mse-known-mu-sigma
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-mle-known-mu-sigma-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0


__units__=(128 64 32 16 8 1)
ensemble_array=()
__units__array=()

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


function run {
    __sigma__=$1
    __nb_plays__=$2
    __units__=$3
    sigma=$4
    ensemble=$5
    host_name=`hostname`

    echo "RUN MSE with known mu and sigma, __sigma__: ${__sigma__}, sigma: ${sigma}, __nb_plays__: ${__nb_plays__}, __units__: ${__units__}, ensemble: ${ensemble}, hostname: ${host_name}"

    source /home/zxchen/.venv3/bin/activate
    # python run_hnn_mse.py --epochs 1000  --mu 0 --sigma 0 --lr 0.05 --points 1000 --nb_plays 50 --units 50 --method debug-mc --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --diff-weights --ensemble ${ensemble}
}

run ${__sigma__array[SLURM_ARRAY_TASK_ID]} ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${sigma_array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
