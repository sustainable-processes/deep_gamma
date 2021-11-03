#!/bin/bash

sweep_id=${sweep_id:uue071bi}

while [ $# -gt 0 ]; do

    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi

    shift
done

wandb login eddd91debd4aeb24f212695d6c663f504fdb7e3c
wandb agent -p vle -e ceb-sre $sweep_id
