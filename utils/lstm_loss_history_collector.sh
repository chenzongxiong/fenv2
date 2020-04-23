diff_weights=1
sigma=110
markov_chain=1
__activation__=tanh

for nb_plays in 20
do
    units=$nb_plays
    if [[ $nb_plays == 500 ]]; then
        units=100
    fi
    for __units__ in 1 8 16 32 64 128 256
    do
        if [[ $markov_chain == 1 ]]; then
            python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma ${sigma} --mu 0 --lr 0.005 --__units__ ${__units__} --activation None --diff-weights --markov-chain --__activation__ ${__activation__}
        elif [[ $diff_weights == 1 ]]; then
            python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma ${sigma} --mu 0 --lr 0.005 --__units__ ${__units__} --activation tanh --diff-weights &
        else
            python utils/lstm_filter_loss.py --units ${units} --nb_plays ${nb_plays} --points 1000 --sigma ${sigma} --mu 0 --lr 0.005 --__units__ ${__units__} --activation tanh &
        fi
    done
done
