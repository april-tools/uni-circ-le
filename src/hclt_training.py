import os
import sys
import argparse
from typing import Literal, Optional
import torch

import functools
print = functools.partial(print, flush=True)

sys.path.append(os.path.join(os.getcwd(), "cirkit"))
sys.path.append(os.path.join(os.getcwd(), "src"))

from torch.utils.tensorboard import SummaryWriter
from utils import *
from measures import *
from tqdm import tqdm
import time


# cirkit
from cirkit.models.functional import integrate
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.reparams.leaf import ReparamExp, ReparamIdentity, ReparamLeaf, ReparamSoftmax
from cirkit.layers.input.exp_family.categorical import CategoricalLayer
from cirkit.layers.input.exp_family.binomial import BinomialLayer
from cirkit.layers.sum_product import CollapsedCPLayer, TuckerLayer, SharedCPLayer
from cirkit.models.tensorized_circuit import TensorizedPC
from cirkit.region_graph import RegionGraph, RegionNode, PartitionNode
from cirkit.region_graph.poon_domingos import PoonDomingos
from cirkit.region_graph.quad_tree import QuadTree


class ReparamReLU(ReparamLeaf):
    eps = torch.finfo(torch.get_default_dtype()).tiny
    def forward(self) -> torch.Tensor:
        return torch.clamp(self.param, min=ReparamReLU.eps)

class ReparamSoftplus(ReparamLeaf):
    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.param)

LAYER_TYPES = {
    "tucker": TuckerLayer,
    "cp": CollapsedCPLayer,
    "cp-shared": SharedCPLayer,
}
RG_TYPES = {"QG", "QT", "PD"}
LEAF_TYPES = {"cat": CategoricalLayer, "bin": BinomialLayer}
REPARAM_TYPES = {
    "exp": ReparamExp,
    "relu": ReparamReLU,
    "softplus": ReparamSoftplus,
    "clamp": ReparamIdentity,
    "softmax": ReparamSoftmax
    # "exp_temp" will be added at run-time
    # "softmax_temp" will be added at run-time
}

def tree2rg(tree: np.ndarray) -> RegionGraph:

    num_variables = len(tree)
    rg = RegionGraph()
    partitions: List[Optional[PartitionNode]] = [None] * num_variables
    for v in range(num_variables):
        cur_v, prev_v = v, tree[v]
        while prev_v != -1:
            if partitions[prev_v] is None:
                p_scope = {v, prev_v}
                partition_node = PartitionNode(p_scope)
                partitions[prev_v] = partition_node
            else:
                p_scope = set(partitions[prev_v].scope)
                p_scope = {v} | p_scope
                partition_node = PartitionNode(p_scope)
                partitions[prev_v] = partition_node
            cur_v, prev_v = prev_v, tree[cur_v]

    regions: List[Optional[RegionNode]] = [None] * num_variables
    for cur_v in range(num_variables):
        prev_v = tree[cur_v]
        leaf_region = RegionNode({cur_v})
        if partitions[cur_v] is None:
            if prev_v != -1:
                rg.add_edge(leaf_region, partitions[prev_v])
            regions[cur_v] = leaf_region
        else:
            rg.add_edge(leaf_region, partitions[cur_v])
            p_scope = partitions[cur_v].scope
            if regions[cur_v] is None:
                regions[cur_v] = RegionNode(set(p_scope))
            rg.add_edge(partitions[cur_v], regions[cur_v])
            if prev_v != -1:
                rg.add_edge(regions[cur_v], partitions[prev_v])

    return rg


def train_procedure(
    *,
    pc: TensorizedPC,
    pc_hypar: dict,
    dataset_name: str,
    save_path: str,
    batch_size: int = 128,
    lr: float = 0.01,
    max_num_epochs: int = 200,
    patience: int = 10,
    verbose: bool = True,
    compute_train_ll: bool = False
):
    """Train a TensorizedPC using gradient descent.

    Args:
        pc (TensorizedPC): prob. circ. to train
        dataset_name (str): dataset name
        save_path (str): path (including file name and extension) for saving the trained model
        tensorboard_dir (str): directory where to write tensorboard logs
        model_name (str): name for the model in tensorbboard
        rg_name (str): Name
        layer_used (str): _description_
        leaf_type (str): _description_
        k (int): _description_
        k_in (int): _description_
        batch_size (int, optional): _description_. Defaults to 100.
        lr (float, optional): _description_. Defaults to 0.01.
        max_num_epochs (int, optional): _description_. Defaults to 200.
        patience (int, optional): _description_. Defaults to 3.
        verbose (bool, optional): _description_. Defaults to True.
    """
    pc_pf: TensorizedPC = integrate(pc)
    torch.set_default_tensor_type("torch.FloatTensor")

    # make experiment name string
    exp_name = "_".join([pc_hypar["DATA"], pc_hypar["RG"], pc_hypar["PAR"], pc_hypar["LEAF"],
                         f"K_{pc_hypar['K']}", f"KIN_{pc_hypar['K_IN']}", f"lr_{pc_hypar['lr']}"])

    # model id uses date_time
    model_id: str = os.path.splitext(os.path.basename(save_path))[0]
    exp_name = f"{exp_name}_{model_id}"
    print("Experiment name: " + exp_name)

    # load data
    x_train, x_valid, x_test = load_dataset(dataset_name, device=get_pc_device(pc))
    # load optimizer
    optimizer = torch.optim.Adam(pc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.eta_min)

    # Setup Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(save_path), model_id))

    # maybe creates save dir
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    def print_ll(tr_ll: float, val_ll: float, text: str):
        print(text, end="")
        print(f"\ttrain LL {tr_ll}", end="")
        print(f"\tvalid LL {val_ll}")


    """
    # Compute train and validation log-likelihood
     train_ll = eval_loglikelihood_batched(pc, x_train, device=device) / x_train.shape[0] if compute_train_ll else np.NAN
     valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]

    print_ll(train_ll, valid_ll, "[Before Learning]")

    # ll on tensorboard
    if not np.isnan(train_ll):
        writer.add_scalar("train_ll", train_ll, 0)
    writer.add_scalar("valid_ll", valid_ll, 0)
    """

    # # # # # # # # # # # #
    # SETUP Early Stopping
    best_valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]
    # best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]
    patience_counter = patience

    tik_train = time.time()
    for epoch_count in range(1, max_num_epochs + 1):

        # # # # # #
        #  Train  #
        # # # # # #
        idx_batches = torch.randperm(x_train.shape[0]).split(batch_size)

        # setup tqdm
        if verbose:
            pbar = tqdm(
                iterable=enumerate(idx_batches),
                total=len(idx_batches),
                unit="steps",
                ascii=" ▖▘▝▗▚▞█",
                ncols=120,
                )
        else:
            pbar = enumerate(idx_batches)

        train_ll = 0
        for batch_count, idx in pbar:
            batch_x = x_train[idx, :].unsqueeze(dim=-1).to(device)

            if batch_size == 1:
                batch_x = batch_x.reshape(1, -1)

            log_likelihood = (pc(batch_x) - pc_pf(batch_x)).sum(0)
            objective = -log_likelihood
            optimizer.zero_grad()
            objective.backward()

            # update with batch ll
            train_ll += log_likelihood

            # CHECK
            check_validity_params(pc)
            # UPDATE
            optimizer.step()
            scheduler.step()

            # CHECK AGAIN
            check_validity_params(pc)

            # project params in inner layers TODO: remove or edit?
            if pc_hypar["REPARAM"] == "clamp":
                for layer in pc.inner_layers:
                    if type(layer) == CollapsedCPLayer:
                        layer.params_in().data = torch.clamp(layer.params_in(), min=np.sqrt(ReparamReLU.eps))
                    else:
                        layer.params().data = torch.clamp(layer.params(), min=np.sqrt(ReparamReLU.eps))

            if verbose:
                if batch_count % 10 == 0:
                    pbar.set_description(f"Epoch {epoch_count} Train LL={objective.item() / batch_size :.2f})")

        train_ll = train_ll / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid, device=device) / x_valid.shape[0]

        print_ll(train_ll, valid_ll, f"[After epoch {epoch_count}]")
        if device != "cpu": print('Max allocated GPU: %.2f' % (torch.cuda.max_memory_allocated() / 1024 ** 3))

        # Not improved
        if valid_ll <= best_valid_ll:
            patience_counter -= 1
            if patience_counter == 0:
                print("-> Validation LL did not improve, early stopping")
                break

        else:
            # Improved, save model
            torch.save(pc, save_path)
            print("-> Saved model")

            # update best_valid_ll
            best_valid_ll = valid_ll
            # best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]
            patience_counter = patience

        writer.add_scalar("train_ll", train_ll, epoch_count)
        writer.add_scalar("valid_ll", valid_ll, epoch_count)
        writer.flush()

    print('Overall training time: %.2f (s)', time.time() - tik_train)
    # reload the model and compute test_ll
    pc = torch.load(save_path)
    best_train_ll = eval_loglikelihood_batched(pc, x_train, device=device) / x_train.shape[0]
    best_test_ll = eval_loglikelihood_batched(pc, x_test, device=device) / x_test.shape[0]

    print(args.reparam)
    print('train bpd: ', bpd_from_ll(pc, best_train_ll))
    print('valid bpd: ', bpd_from_ll(pc, best_valid_ll))
    print('test  bpd: ', bpd_from_ll(pc, best_test_ll))

    writer.add_hparams(
        hparam_dict=pc_hypar,
        metric_dict={
            "Best/Valid/ll": float(best_valid_ll),
            "Best/Valid/bpd": float(bpd_from_ll(pc, best_valid_ll)),
            "Best/Test/ll": float(best_test_ll),
            "Best/Test/bpd": float(bpd_from_ll(pc, best_test_ll)),
        },
        hparam_domain_discrete={
            "DATA": ["mnist", "fashion_mnist", "celeba"],
            "RG": ["QG", "PD", "QT"],
            "PAR": ["cp", "cpshared", "tucker"],
            "LEAF": ["bin", "cat"],
            "REPARAM": ["softplus", "exp", "exp_temp", "relu"]
        },
    )
    writer.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser("MNIST experiments.")
    parser.add_argument("--seed",           type=int,   default=42,             help="Random seed")
    parser.add_argument("--gpu",            type=int,   default=None,           help="Device on which run the benchmark")
    parser.add_argument("--dataset",        type=str,   default="mnist",        help="Dataset for the experiment")
    parser.add_argument("--model-dir",      type=str,   default="out",          help="Base dir for saving the model")
    parser.add_argument("--lr",             type=float, default=0.1,            help="Path of the model to be loaded")
    parser.add_argument("--num-sums",       type=int,   default=128,            help="Num sums")
    parser.add_argument("--num-input",      type=int,   default=None,           help="Num input distributions per leaf, if None then is equal to num-sums",)
    parser.add_argument("--layer",          type=str,                           help="Layer type: 'tucker', 'cp' or 'cp-shared'")
    parser.add_argument("--leaf",           type=str,                           help="Leaf type: either 'cat' or 'bin'")
    parser.add_argument("--reparam",        type=str,   default="exp",          help="Either 'exp', 'relu', or 'exp_temp'")
    parser.add_argument("--max-num-epochs", type=int,   default=200,            help="Max num epoch")
    parser.add_argument("--batch-size",     type=int,   default=128,            help="batch size")
    parser.add_argument("--train-ll",       type=bool,  default=False,          help="Compute train-ll at the end of each epoch")
    parser.add_argument("--progressbar",    type=bool,  default=False,          help="Print the progress bar")
    parser.add_argument('-t0',              type=int,   default=1,              help='sched CAWR t0, 1 for fixed lr ')
    parser.add_argument('-eta_min',         type=float, default=1e-4,           help='sched CAWR eta min')
    args = parser.parse_args()
    print(args)
    init_random_seeds(seed=args.seed)

    device = (
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu is not None
        else "cpu"
    )
    assert args.layer in LAYER_TYPES
    assert args.leaf in LEAF_TYPES
    if args.num_input is None:
        args.num_input = args.num_sums

    train_x, valid_x, test_x = load_dataset(args.dataset, device="cpu")

    # Setup region graph
    rg = tree2rg(np.array([28, 0, 3, 31, 5, 6, 7, 35, 7, 8, 38, 39, 40,
                41, 15, 43, 44, 45, 17, 47, 48, 49, 23, 51, 25, 53,
                54, 55, 56, 57, 31, 59, 60, 32, 62, 63, 8, 38, 39,
                40, 41, 42, 43, 44, 72, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 57, 58, 59, 60, 88, 89, 61, 62, 63,
                93, 38, 39, 40, 41, 42, 43, 100, 45, 46, 47, 48, 49,
                50, 78, 81, 53, 54, 55, 56, 57, 58, 86, 89, 117, 62,
                63, 64, 121, 66, 67, 68, 69, 70, 71, 128, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 83, 140, 141, 142, 87, 144,
                145, 146, 147, 148, 149, 150, 151, 152, 153, 125, 126, 127, 128,
                129, 132, 133, 134, 135, 107, 135, 136, 139, 111, 141, 142, 143,
                115, 145, 146, 147, 148, 149, 150, 151, 152, 180, 152, 126, 127,
                128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                141, 142, 170, 173, 145, 146, 147, 148, 149, 150, 151, 208, 209,
                210, 211, 212, 213, 214, 215, 216, 217, 191, 219, 220, 192, 222,
                223, 224, 225, 199, 227, 228, 200, 230, 231, 232, 233, 234, 208,
                209, 210, 238, 212, 213, 241, 213, 243, 244, 245, 246, 218, 219,
                220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
                235, 207, 208, 238, 239, 211, 239, 242, 243, 244, 245, 273, 245,
                246, 249, 221, 222, 223, 224, 225, 255, 227, 257, 229, 230, 231,
                232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
                301, 246, 274, 277, 278, 279, 251, 281, 282, 283, 284, 285, 257,
                285, 286, 289, 261, 289, 290, 293, 265, 293, 294, 297, 298, 299,
                300, 328, 300, 301, 302, 305, 277, 278, 279, 280, 281, 311, 283,
                313, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
                297, 298, 299, 356, 301, 302, 330, 304, 332, 333, 334, 337, 309,
                310, 338, 341, 313, 341, 342, 372, 373, 318, 319, 349, 350, 322,
                323, 351, 352, 382, 383, 355, 356, 357, 358, 361, 333, 334, 391,
                392, 366, 367, 368, 369, 341, 342, 372, 373, 401, 373, 374, 404,
                349, 350, 407, 408, 382, 383, -1, 356, 357, 414, 386, 389, 417,
                418, 419, 393, 421, 395, 367, 368, 396, 370, 427, 428, 429, 430,
                431, 432, 433, 434, 435, 436, 437, 382, 439, 440, 441, 415, 443,
                444, 416, 446, 447, 448, 449, 423, 451, 452, 424, 427, 455, 427,
                430, 458, 430, 431, 461, 435, 436, 437, 438, 410, 438, 439, 469,
                443, 471, 445, 473, 474, 475, 476, 477, 451, 479, 480, 452, 482,
                483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 437, 494, 495,
                469, 497, 498, 470, 471, 472, 473, 474, 477, 478, 479, 480, 481,
                482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 465,
                493, 494, 495, 496, 497, 498, 499, 500, 474, 475, 476, 477, 507,
                479, 480, 508, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
                492, 493, 494, 495, 496, 497, 498, 526, 500, 528, 558, 559, 560,
                561, 562, 534, 564, 536, 566, 567, 568, 569, 570, 571, 572, 573,
                574, 575, 576, 577, 578, 579, 580, 581, 555, 527, 584, 556, 586,
                558, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
                573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 554, 582, 583,
                584, 585, 559, 560, 561, 562, 590, 564, 592, 566, 567, 568, 569,
                570, 571, 572, 573, 574, 575, 576, 577, 578, 635, 636, 581, 582,
                610, 584, 612, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651,
                652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 606, 663, 664,
                665, 639, 667, 668, 640, 670, 671, 672, 673, 647, 675, 649, 677,
                678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 634,
                691, 692, 693, 638, 695, 667, 641, 669, 670, 673, 674, 675, 676,
                677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689,
                690, 662, 690, 691, 692, 695, 723, 697, 669, 670, 671, 672, 673,
                674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686,
                687, 688, 689, 690, 691, 692, 693, 694, 751, 696, 697, 698, 699,
                700, 701, 731, 703, 704, 732, 706, 707, 708, 709, 710, 767, 768,
                769, 770, 771, 772, 773, 774, 748, 749, 721, 749, 750, 751, 752,
                755, 727, 757, 758, 730, 758, 761, 733, 761, 762, 763, 766, 738,
                766, 767, 768, 769, 770, 771, 772, 773, 774, 748, 749, 750, 778,
                752, 780, 754, 755]))

    # Setup leaves setting
    efamily_kwargs: dict
    if args.leaf == "cat":
        efamily_kwargs = {"num_categories": 256}
    elif args.leaf == "bin":
        efamily_kwargs = {"n": 256}

    # setup reparam
    class ReparamExpTemp(ReparamLeaf):
        def forward(self) -> torch.Tensor:
            return torch.exp(self.param / np.sqrt(args.num_sums))

    class ReparamSoftmaxTemp(ReparamLeaf):
        def forward(self) -> torch.Tensor:
            param = self.param if self.log_mask is None else self.param + self.log_mask
            param = self._unflatten_dims(torch.softmax(self._flatten_dims(param) / np.sqrt(args.num_sums),
                                                       dim=self.dims[0]))
            return torch.nan_to_num(param, nan=1)

    REPARAM_TYPES["exp_temp"] = ReparamExpTemp
    REPARAM_TYPES["softmax_temp"] = ReparamSoftmaxTemp

    # Create probabilistic circuit
    pc = TensorizedPC.from_region_graph(
        rg=rg,
        layer_cls=LAYER_TYPES[args.layer],
        efamily_cls=LEAF_TYPES[args.leaf],
        efamily_kwargs=efamily_kwargs,
        num_inner_units=args.num_sums,
        num_input_units=args.num_input,
        reparam=REPARAM_TYPES[args.reparam],
    )
    pc.to(device)
    print(f"Num of params: {num_of_params(pc)}")
    model_name = args.layer

    # compose model path
    # e.g. out/mnist/
    save_path = os.path.join(
        args.model_dir,
        args.dataset,
        'HCLT',
        args.layer,
        args.leaf,
        args.reparam,
        str(args.num_sums),
        str(args.lr),
        get_date_time_str() + ".mdl",
    )

    # Train the model
    train_procedure(
        pc=pc,
        pc_hypar={
            "RG": 'HCLT',
            "PAR": args.layer,
            "LEAF": args.leaf,
            "REPARAM": args.reparam,
            "K": args.num_sums,
            "K_IN": args.num_input,
            "DATA": args.dataset,
            "lr": args.lr,
            "optimizer": "Adam",
            "batch_size": args.batch_size,
            "num_par": num_of_params(pc)
            },
        dataset_name=args.dataset,
        save_path=save_path,
        batch_size=args.batch_size,
        lr=args.lr,
        max_num_epochs=args.max_num_epochs,
        patience=10,
        compute_train_ll=args.train_ll,
        verbose=args.progressbar
    )
