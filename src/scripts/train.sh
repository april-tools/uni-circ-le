python src/training.py --dataset "mnist" --model-dir "out/test_mixing" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --freeze-mixing-layers "not_last" --gpu 1
python src/training.py --dataset "mnist" --model-dir "out/test_mixing" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --freeze-mixing-layers "all" --gpu 0

python src/training.py --dataset "fashion_mnist" --model-dir "out/test_mixing" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --freeze-mixing-layers "not_last" --gpu 3
python src/training.py --dataset "cifar10" --model-dir "out/cifar10" --lr 0.01 --k 256  --rg "QT" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 4
