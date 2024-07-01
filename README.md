# uni-circ-le

## Training

Some examples for training experiment
```bash
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 64  --rg "QG" --layer "tucker" --batch-size 256 --max-num-epochs 200 --gpu 0
    python src/training.py --dataset "fashion_mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0

```