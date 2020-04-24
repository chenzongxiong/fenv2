python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 0 --sigma 1 --method mc --diff-weights --activation tanh --with-noise
python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 0 --sigma 5 --method mc --diff-weights --activation tanh --with-noise

python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 1 --sigma 1 --method mc --diff-weights --activation tanh --with-noise
python dataset_generator.py --model-noise --points 1000 --nb_plays 50 --units 50 --mu 1 --sigma 5 --method mc --diff-weights --activation tanh --with-noise
