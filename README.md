# uni-circ-le

## Training

Some examples for training experiment
```bash
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 64  --rg "QG" --layer "tucker" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
    python src/training.py --dataset "fashion_mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
```

The model is saved as in the following example
`out/training/models/mnist/QG/cp/cat/clamp/k_512/lr_0.01/b_256/<date_time>.mdl`

Note the folder hierarchy:
1. dataset
2. region graph
3. parameterization
4. leaves type (always `cat` in our experiments), for categorical leaves
5. weights re-parameterization (always `clamp` in our experiments)
6. hyperparameter `K`
7. learning rate
8. batch size

## Compression
To compress a model parameterized with `Tucker` layers:
```bash
    python src/compression.py --dataset "mnist" --tucker-model-path "<path_to_tucker_model>" --save-model-dir "out/compressed/mnist/example_QG" --rg "QG" --input-type "cat" --rank 1 --gpu 0
```
The _compressed_ model is saved as in the following path (basically the `--save-model-dir` followed by `rank_<rank>.mdl`):
`out/compressed/mnist/example_QG/rank_1.mdl`

## Train a compressed model
To train an existing model (e.g. a compressed one) use the following script:
```bash
    python src/finetuning.py --dataset mnist --model-path "out/compressed/mnist/example_QG/rank_1.mdl" --lr 0.01 --rg QG --rank 1 --max-num-epochs 200 --batch-size 256 --progressbar True --gpu 0
```

The fine-tuned compressed model is saved in the same folder of the starting model, but the file is named as, 
in this case, `rank_1finetuned.mdl`.