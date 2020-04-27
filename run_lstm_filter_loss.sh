#!/bin/bash

__units__LIST=()
ensemble_LIST=()

for __units__ in 1 8 16 32 64 128
do
    for seq in {1..20}
    do
        __units__LIST+=($__units__)
        ensemble_LIST+=($seq)
    done
done

for seq in {40..59}
do
    echo "Colleting __units__=${__units__LIST[seq]} in seq ${seq}, ensemble is: ${ensemble_LIST[seq]}"
    python utils/lstm_filter_loss.py --mu 0 --sigma 0 --diff-weights --__units__ ${__units__LIST[seq]} --method debug-dima --ensemble ${ensemble_LIST[seq]} --seq ${seq}
    python utils/lstm_filter_loss.py --mu 0 --sigma 0 --diff-weights --__units__ ${__units__LIST[seq]} --method debug-pavel --ensemble ${ensemble_LIST[seq]} --seq ${seq}
done
