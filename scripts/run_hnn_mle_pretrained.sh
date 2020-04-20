for i in 1 2 3 4 5 6 7 10 11 12 13 14 15 16 17 18 19
do
    python run_hnn_mle.py --__nb_plays__ 100 --__units__ 100 --__activation__ elu --batch_size 1500 --ensemble ${i}
done
