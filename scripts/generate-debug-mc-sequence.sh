#!/bin/bash

#SBATCH -J generate-debug-mc-sequence
#SBATCH -D /home/zxchen/fenv2
#SBATCH -o ./tmp/generate-debug-mc-sequence.log
#SBATCH --nodes=1
#SBATCH --constraint "AMD"
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --partition=big
#SBATCH --mail-type=end
#SBATCH --mail-user=czxczf@gmail.com

# python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 0 --sigma 1 --method mc --diff-weights --activation tanh --with-noise
python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 0 --sigma 5 --method mc --diff-weights --activation tanh --with-noise

# python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 1 --sigma 1 --method mc --diff-weights --activation tanh --with-noise
# python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 1 --sigma 5 --method mc --diff-weights --activation tanh --with-noise
