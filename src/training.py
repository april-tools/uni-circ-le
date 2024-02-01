import os
import sys
sys.path.append("../cirkit")

from cirkit.new.model.functional import integrate
from cirkit.new.model.tensorized_circuit import TensorizedCircuit
from torch.utils.tensorboard import SummaryWriter
import utils
# from cirkit.models.tensorized_circuit import TensorizedPC
from measures import *



def train_procedure(
    pc: TensorizedCircuit,
    dataset_name: str,
    model_dir: str,
    tensorboard_dir: str,
    model_name: str,
    rg_name: str,
    layer_used: str,
    k: int,
    k_in: int,
    prod_exp: bool,
    batch_size=100,
    lr=0.01,
    max_num_epochs=200,
    patience=3,
    verbose=True,
):
    
    pc_pf = integrate(pc)
    torch.set_default_tensor_type("torch.FloatTensor")

    # make experiment name string
    hyperparams = {
        "rg": rg_name,
        "layer": layer_used,
        "dataset": dataset_name,
        "lr": lr,
        "optimizer": "Adam",
        "batch_size": batch_size,
        "num_parameters": utils.num_of_params(pc),
        "k": k,
        "k_in": k_in,
        "prod_exp": prod_exp,
        "patience": patience,
    }

    hyp_in_name = {
        k: hyperparams[k] for k in {"dataset", "rg", "layer", "k", "k_in", "lr"}
    }
    experiment_name = "".join([f"{key}_{value}_" for key, value in hyp_in_name.items()])
    experiment_name += "".join(
        [
            f"{key}_{value:.3f}_"
            for key, value in hyp_in_name.items()
            if type(value) == float
        ]
    )
    experiment_name = (
        f"{model_name}_" + experiment_name + f"_{utils.get_date_time_str()}"
    )
    assert experiment_name != ""
    print("Experiment name: " + experiment_name)

    model_path = model_dir + "/" + experiment_name + ".mdl"

    x_train, x_valid, x_test = utils.load_dataset(
        dataset_name, device=utils.get_pc_device(pc)
    )

    optimizer = torch.optim.Adam(pc.parameters(), lr=lr)  # check if is correct

    if tensorboard_dir is None:
        tensorboard_dir = "runs"
    writer = SummaryWriter(
        log_dir=os.path.join(os.getcwd(), f"{tensorboard_dir}/{experiment_name}")
    )
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
    valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

    if verbose:
        print("[Before Learning]")
        print("\ttrain LL {}   valid LL {}".format(train_ll, valid_ll))

    # ll on tensorboard
    writer.add_scalar("train_ll", train_ll, 0)
    writer.add_scalar("valid_ll", valid_ll, 0)

    ####
    # SETUP
    best_valid_ll = valid_ll
    best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
    patience_counter = patience
    pc_scope = [i for i in range(pc.num_variables)]
    for epoch_count in range(1, max_num_epochs + 1):

        # # # # # #
        #   LEARN
        # # # # # #
        idx_batches = torch.randperm(x_train.shape[0]).split(batch_size)
        for batch_count, idx in enumerate(idx_batches):
            batch_x = x_train[idx, :]

            if batch_size == 1:
                batch_x = batch_x.reshape(1, -1)

            log_likelihood = (pc(batch_x) - pc_pf()).sum(0)
            objective = -log_likelihood
            optimizer.zero_grad()
            objective.backward()

            # CHECK
            utils.check_validity_params(pc)
            # UPDATE
            optimizer.step()
            # CHECK AGAIN
            utils.check_validity_params(pc)

            # project params in inner layers
            for layer in pc.inner_layers:
                layer.clamp_params()

        train_ll = eval_loglikelihood_batched(pc, x_train) / x_train.shape[0]
        valid_ll = eval_loglikelihood_batched(pc, x_valid) / x_valid.shape[0]

        if verbose:
            print(f"[After epoch {epoch_count}]")
            print("\ttrain LL {}   valid LL {}".format(train_ll, valid_ll))

        # Not improved
        if valid_ll <= best_valid_ll:
            patience_counter -= 1
            if patience_counter == 0:
                if verbose:
                    print("-> Validation LL did not improve, early stopping")
                break

        else:
            # Improved, save model
            torch.save(pc, model_path)
            if verbose:
                print("-> Saved model")

            # update best_valid_ll
            best_valid_ll = valid_ll
            best_test_ll = eval_loglikelihood_batched(pc, x_test) / x_test.shape[0]
            patience_counter = patience

        writer.add_scalar("train_ll", train_ll, epoch_count)
        writer.add_scalar("valid_ll", valid_ll, epoch_count)
        writer.flush()

    writer.add_hparams(
        hparam_dict=hyperparams,
        metric_dict={
            "Best/Valid/ll": best_valid_ll,
            "Best/Valid/bpd": bpd_from_ll(pc, best_valid_ll),
            "Best/Test/ll": float(best_test_ll),
            "Best/Test/bpd": float(bpd_from_ll(pc, best_test_ll)),
        },
        hparam_domain_discrete={
            "dataset": ["mnist", "fashion_mnist"],
            "rg": ["QG", "PD", "QT"],
            "layer": ["cp", "cp-shared", "tucker"]
        },
    )
    writer.close()
