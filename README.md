# uni-circ-le

## Early Experiments to run

We want to check the difference in performance with the various re-parameterizations for mixture weights.
The re-parameterizations here used are:
- `exp`: $\exp(\theta)$
- `exp-temp`: $\exp(\frac{\theta}{\sqrt{K}}) $
- `relu`: $\max(\varepsilon, \theta)$
- `softplus`: $\log(1 + \exp(\theta))$
- `clamp`: projected gradient descent (clamping after each optimizer update $\max(\varepsilon, \theta)$ )

First, we train the best performing model (fig. 4a of the workshop paper)
```bash
    for repar in exp exp-temp relu softplus clamp
    do
        python src/training.py --dataset "mnist" --model-dir "out/" \
            --rg "QG" --layer "cp" --leaf "cat" --reparam repar --num-sums 256 \
            --lr 0.01 --batch-size 128 --max-num-epochs 500
    done
```
Results (statistics computed over three runs):

| repar    | Train_mean | Train_std | Valid_mean | Valid_std | Test_mean | Test_std |
|----------|------------|-----------|------------|-----------|-----------|----------|
| clamp    | 1.097      | 0.006     | 1.212      | 0.001     | 1.174     | 0.001    |
| exp      | 1.186      | 0.010     | 1.292      | 0.002     | 1.253     | 0.002    |
| exp_temp | 1.150      | 0.004     | 1.273      | 0.001     | 1.232     | 0.000    |
| relu     | 1.250      | 0.007     | 1.333      | 0.013     | 1.294     | 0.013    |
| softplus | 1.132      | 0.005     | 1.254      | 0.003     | 1.213     | 0.003    |
