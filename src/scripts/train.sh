python src/training.py --dataset "mnist" \
    --model-dir "out/" \
    --lr 0.01 --num-sums 4  --rg "QG" --layer "cp" --leaf "bin" \
    --batch-size 100 --max-num-epochs 100
