# uni-circ-le

## Training

Some examples for training experiment
```bash
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 64  --rg "QG" --layer "tucker" --batch-size 256 --max-num-epochs 200 --gpu 0
    python src/training.py --dataset "fashion_mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0

```

To test 

Results (statistics computed over three runs):

| repar    | Train_mean | Train_std | Valid_mean | Valid_std | Test_mean | Test_std |
|----------|------------|-----------|------------|-----------|-----------|----------|
| clamp    | 1.097      | 0.006     | 1.212      | 0.001     | 1.174     | 0.001    |
| exp      | 1.186      | 0.010     | 1.292      | 0.002     | 1.253     | 0.002    |
| exp_temp | 1.153      | 0.005     | 1.273      | 0.002     | 1.231     | 0.001    |
| relu     | 1.250      | 0.007     | 1.333      | 0.013     | 1.294     | 0.013    |
| softplus | 1.132      | 0.005     | 1.254      | 0.003     | 1.213     | 0.003    |
