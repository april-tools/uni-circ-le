python src/training.py --dataset "mnist" \
    --model-dir "out/" \
    --lr 0.01 --num-sums 4  --rg "QG" --layer "cp" --leaf "cat" --reparam "softplus" \
    --batch-size 128 --max-num-epochs 5 --progressbar True
