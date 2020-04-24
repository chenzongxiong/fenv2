#!/bin/bash

#SBATCH -J run-hnn-mle
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/run-hnn-mle-%a.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com
#SBATCH --array=0

# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 2000 --ensemble 1 --force-train &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-25-__units__-25-__activation__-elu-points-2000.log
# python run_hnn_mle.py --__nb_plays__ 50 --__units__ 50 --__activation__ elu --batch_size 2000 --ensemble 1 --force-train &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-50-__units__-50-__activation__-elu-points-2000.log
# python run_hnn_mle.py --__nb_plays__ 50 --__units__ 100 --__activation__ elu --batch_size 2000 --ensemble 1 --force-train &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-50-__units__-100-__activation__-elu-points-2000.log
# python run_hnn_mle.py --__nb_plays__ 100 --__units__ 100 --__activation__ elu --batch_size 2000 --ensemble 1 --force-train &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-100-__units__-100-__activation__-elu-points-2000.log
# python run_hnn_mle.py --__nb_plays__ 200 --__units__ 100 --__activation__ elu --batch_size 2000 --ensemble 1 --force-train &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-200-__units__-100-__activation__-elu-points-2000.log

# experiments for non-fixed mu
# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 2000 --ensemble 1000 --force-train --learnable-mu &> log/hnn-activation-None-lr-0.07-mu-0-sigma-110-nb_play-20-units-10000-__nb_plays__-25-__units__-25-__activation__-elu-points-2000-ensembel-1000.log

python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 1 --mu 0 --sigma 1 --method mc
# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 0 --__sigma__ 5 --mu 0 --sigma 5

# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 1 --__sigma__ 1 --mu 0 --sigma 1
# python run_hnn_mle.py --__nb_plays__ 25 --__units__ 25 --__activation__ elu --batch_size 1000 --ensemble 1 --force-train --__mu__ 1 --__sigma__ 5 --mu 0 --sigma 5
