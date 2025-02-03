# uni-circ-le

This is the official repository containing the code to reproduce the experiments shown in
[_"What is the Relationship between Tensor Factorizations and Circuits (and How Can We Exploit it)?"_](https://arxiv.org/abs/2409.07953),
which has been accepted in TMLR with the featured certification.

**Abstract:**
We establish a rigorous connection between circuit representations and tensor factorizations.
By doing so, we highlight a series of opportunities that can benefit both communities.
Our work generalizes popular tensor factorizations within the circuit language, and unifies various circuit learning algorithms under a single factorization framework.
In turn, this allows us to systematically construct and explore various circuit and tensor factorization models while maintaining tractability.
We show the effectiveness of our framework through extensive empirical evaluations, and highlight new research opportunities for tensor factorizations in probabilistic modeling. 

> [!NOTE]
> The current implementation of tensorized circuits in this repository is based on [_ten-pcs_](https://github.com/april-tools/ten-pcs),
> which has been deprecated in favour of [_cirkit_](https://github.com/april-tools/cirkit),
> a general-purpose circuit framework to build, learn and reason about probabilistic models that is based on PyTorch.
> The implementation of the pipeline we describe in the paper to build tensorized circuits is entirely supported by
> [_cirkit_](https://github.com/april-tools/cirkit).

## Training Circuits

Some examples to train a Probabilistic Circuit
```bash
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
    python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 64  --rg "QG" --layer "tucker" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
    python src/training.py --dataset "fashion_mnist" --model-dir "out/training" --lr 0.01 --k 512  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 200 --gpu 0 --progressbar True
```
Useful parameters:
- `dataset`: use `mnist`, `fashion_mnist` or `celeba`
- `rg`: the region graph to use. Available options are `PD` (Poon Domingo's), `QG` (Quad Graph) and `QT` (Quad Tree).
- `layer`: the parameterization (or layer) to use. Available options are `tucker`, `cp`, `cp-s`, `cp-xs`, as in the paper.
- `k`: is the number of 
- `lr`: the best learning rate found for `QG` or `QT` region graph is 0.01, while the best for `PD` region graph is 0.1

### Example to Run Locally

By training the following small CP model (can be easily executed locally on a laptop):
```python src/training.py --dataset "mnist" --model-dir "out/training" --lr 0.01 --k 4  --rg "QG" --layer "cp" --batch-size 256 --max-num-epochs 5 --progressbar True```
The model is saved as in the following example
```out/training/models/mnist/QG/cp/cat/clamp/k_4/lr_0.01/b_256/<date_time>.mdl```

Note the folder hierarchy:
1. dataset
2. region graph
3. parameterization
4. leaves type (always `cat` in our experiments), for categorical leaves
5. weights re-parameterization (always `clamp` in our experiments)
6. hyperparameter `K`
7. learning rate
8. batch size

## Compression of Tucker Layers

To compress a model parameterized with `Tucker` layers:
```bash
    python src/compression.py --dataset "mnist" --tucker-model-path "<path_to_tucker_model>" --save-model-dir "out/compressed/mnist/example_QG" --rg "QG" --input-type "cat" --rank 1 --gpu 0
```
The _compressed_ model is saved as in the following path (basically the `--save-model-dir` followed by `rank_<rank>.mdl`):
`out/compressed/mnist/example_QG/rank_1.mdl`

## Train a Compressed Circuit Model

To train an existing model (e.g. a compressed one, in our example the previous compressed model) use the following script:
```bash
    python src/finetuning.py --dataset mnist --model-path "out/compressed/mnist/example_QG/rank_1.mdl" --lr 0.01 --rg QG --rank 1 --max-num-epochs 200 --batch-size 256 --progressbar True --gpu 0
```

The fine-tuned compressed model is saved in the same folder of the starting model, but the file is named as, 
in this case, `rank_1finetuned.mdl`.
