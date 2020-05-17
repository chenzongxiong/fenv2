#!/bin/bash

#SBATCH -J run-hnn-elu-mle-stock-known-mu-sigma
#SBATCH -D /data/scratch/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-elu-mle-stock-known-mu-sigma-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=30-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=6

__nb_plays__=(100)
__units__=(100)
sigma=(10 20 110)
__sigma__=(10 20 110)

__nb_plays__array=()
__units__array=()
ensemble_array=()
__sigma__array=()
sigma_array=()

for i in {1..20}
do
    for k in {0..2}
    do
        for j in {0..0}
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
    __nb_plays__=$1
    __units__=$2
    __sigma__=$3
    sigma=$4
    ensemble=$5
    host_name=`hostname`

    echo "RUN MLE with known mu and sigma, __sigma__: ${__sigma__}, sigma: ${sigma}, __nb_plays__: ${__nb_plays__}, __units__: ${__units__}, ensemble: ${ensemble}, hostname: ${host_name}"

    source /home/zxchen/.venv3/bin/activate
    python run-hnn-mle-stock.py --__nb_plays__ ${__nb_plays__} --__units__ ${__units__} --__activation__ elu --batch_size 1000 --ensemble ${ensemble} --force-train --__mu__ 0 --__sigma__ ${__sigma__} --mu 0 --sigma ${sigma} --method stock --nb_plays 0 --units 0
}

# __sigma__=0.1
# __nb_plays__=25
# __units__=25
# sigma=0.1
# ensemble=1


run ${__nb_plays__array[SLURM_ARRAY_TASK_ID]} ${__units__array[SLURM_ARRAY_TASK_ID]} ${__sigma__array[SLURM_ARRAY_TASK_ID]} ${sigma_array[SLURM_ARRAY_TASK_ID]} ${ensemble_array[SLURM_ARRAY_TASK_ID]}
# run 100 50 110 110 3
# run 100 50 10 10 3


# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 0.1 --mu 0 --sigma 0.1 --method mc --nb_plays 50 --units 50 --activation elu --learnable-mu
# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 0.5 --mu 0 --sigma 0.1 --method mc --nb_plays 50 --units 50 --activation elu --learnable-mu
