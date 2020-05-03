#!/bin/bash

#SBATCH -J prove_hnn_mse_invertable
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run_prove_hnn_mse_pavel-seq-invertable-%a.log
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=30-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-19


# python prove_hnn_mse_invertable.py --epochs 1000 --diff-weights --activation tanh --mu 0 --sigma 8 --lr 0.1 --points 1000 --nb_plays 50 --units 50 --__nb_plays__ 50 --__units__ 50 --__activation__ tanh --force_train

__nb_plays__=(25)
__units__=(25)

__nb_plays__LIST=()
__units__LIST=()
ensemble_LIST=()
points_LIST=()

for j in {0..0}
do
    for i in {1..20}
    do
        for point in 100 1000 10000
        do
            __nb_plays__LIST+=(${__nb_plays__[j]})
            __units__LIST+=(${__units__[j]})
            ensemble_LIST+=(${i})
            points_LIST+=($points)
        done
    done
done

echo ${__nb_plays__LIST[*]}
echo ${__units__LIST[*]}
echo ${ensemble_LIST[*]}

function run {
    __nb_plays__=$1
    __units__=$2
    ensemble=$3
    host_name=`hostname`
    sigma=0
    points=$4
    # sigma=8
    echo "Run prove hnn invertable with __nb_plays__: ${__nb_plays__}, __units__: ${__units__}, job id: ${SLURM_JOB_ID}, task id: ${SLURM_ARRAY_TASK_ID}, hostname: ${host_name}, ensemble: ${ensemble}, sigma: ${sigma}, points: ${points}"

    source /home/zxchen/.venv3/bin/activate

    # python prove_hnn_mse_invertable.py --epochs 5000 --mu 0 --sigma ${sigma} --lr 0.1 --points 1000 --nb_plays 50 --units 50 --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --force_train --ensemble ${ensemble} --diff-weights --method sin
}

run ${__nb_plays__LIST[SLURM_ARRAY_TASK_ID]} ${__units__LIST[SLURM_ARRAY_TASK_ID]} ${ensemble_LIST[SLURM_ARRAY_TASK_ID]} ${points_LIST[SLURM_ARRAY_TASK_ID]}
