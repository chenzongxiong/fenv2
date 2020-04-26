#!/bin/bash

__units__LIST=()
seq_LIST=()


for seq in {0..119}
do
    for __units__ in 1 8 16 32 64 128
    do
        __units__LIST+=($__units__)
        seq_LIST+=(${seq})
    done
done

for seq in {0..119}
do
    echo "Colleting __units__=${__units__LIST[seq]} in seq ${seq}"
    python utils/lstm_filter_loss.py --mu 0 --sigma 0 --diff-weights --__units__ ${__units__LIST[seq]} --method debug-dima --ensemble 1 --seq ${seq}
    python utils/lstm_filter_loss.py --mu 0 --sigma 0 --diff-weights --__units__ ${__units__LIST[seq]} --method debug-pavel --ensemble 1 --seq ${seq}
done
