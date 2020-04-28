#!/bin/bash

#SBATCH -J run-hnn-mle-known-mu-sigma
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-mle-known-mu-sigma-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0-19

__nb_plays__=(25)
__units__=(25)
__sigma__=(0.1 0.5)
sigma=(0.1 0.5)

__nb_plays__array=()
__units__array=()
ensemble_array=()
__sigma__array=()
sigma_array=()

for k in {0..1}
do
    for j in {0..0}
    do
        for i in {1..20}
        do
            __nb_plays__array+=(${__nb_plays__[j]})
            __units__array+=(${__units__[j]})
            ensemble_array+=($i)
            __sigma__array+=(${__sigma__[k]})
            sigma_array+=(${sigma[k]})
        done
    done
done

echo ${__nb_plays__array[*]}
echo ${__units__array[*]}
echo ${__sigma__array[*]}
echo ${ensemble_array[*]}
echo ${sigma_array[*]}

function run {
    __sigma__=$1
    __nb_plays__=$2
    __units__=$3
    sigma=$4
    ensemble=$5
    host_name=`hostname`

    echo "RUN MLE with known mu and sigma, __sigma__: ${__sigma__}, sigma: ${sigma}, __nb_plays__: ${__nb_plays__}, __units__: ${__units__}, ensemble: ${ensemble}, hostname: ${host_name}"

    source /home/zxchen/.venv3/bin/activate
    # python run_hnn_mle.py --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --batch_size 1000 --ensemble ${ensemble} --force-train --__mu__ 0 --__sigma__ ${__sigma__} --mu 0 --sigma ${sigma} --method mc --nb_plays 50 --units 50 --activation tanh
}

# __sigma__=0.1
# __nb_plays__=25
# __units__=25
# sigma=0.1
# ensemble=1


run ${__sigma__array[SLURM_ARRAY_TASK_ID]} ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${sigma_array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}

# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 0.1 --mu 0 --sigma 0.1 --method mc --nb_plays 50 --units 50 --activation tanh --learnable-mu

# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 0.5 --mu 0 --sigma 0.1 --method mc --nb_plays 50 --units 50 --activation tanh
# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 0.5 --mu 0 --sigma 0.1 --method mc --nb_plays 50 --units 50 --activation tanh --learnable-mu
