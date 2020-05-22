# diff_weights=1
# sigma=110
# markov_chain=1
# __activation__=tanh

units=50
nb_plays=50
__units__array=()
ensemble_array=()

for __units__ in 1 8 16 32 64 128
do
    for ensemble in {1..20}
    do
        __units__array+=(${__units__})
        ensemble_array+=(${ensemble})
    done
done

echo ${ensemble_array[*]}
echo ${__units__array[*]}

for seq in {0..119}
do
    echo "seq: ${seq}, ensemble: ${ensemble_array[seq]}, __units__: ${__units__array[seq]}"

    python utils/rnn_filter_loss.py  --__units__ ${__units__array[seq]} --diff-weights --seq ${seq} --ensemble ${ensemble_array[seq]} --method debug-pavel --mu 0 --sigma 0
    # python utils/rnn_filter_loss.py  --__units__ ${__units__array[seq]} --diff-weights --seq ${seq} --ensemble ${ensemble_array[seq]} --method debug-dima --mu 0 --sigma 0
done


# sigma=0
# for seq in {20..59}
# do
#     ensemble=${ensemble_array[seq]}
#     __units__=${__units__array[seq]}
#     python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma ${sigma} --mu 0 --lr 0.01 --__units__ ${__units__} --diff-weights --seq ${seq} --ensemble ${ensemble} --method debug-dima
# done
