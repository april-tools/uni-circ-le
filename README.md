# uni-circ-le

## Early Experiments to run
We want to check the difference in performance with the various re-parameterizations for mixture weights.
The re-parameterizations here used are:
-

First, we train the best performing model (fig. 4a of the workshop paper)
```bash
    for repar in exp exp-temp relu softplus
    do
        python src/training.py --dataset "mnist" --model-dir "out/" \
            --rg "QG" --layer "cp" --leaf "cat" --reparam repar --num-sums 256 \
            --lr 0.01 --batch-size 128 --max-num-epochs 500
    done
```
