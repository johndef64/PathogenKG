#%%
import gc
import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
import wandb

from src.utils import set_seed
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn
from train_and_eval import (
    get_dataset,
    train,
    test,
    negative_sampling_filtered,
)

# WandB configuration
PROJECT_NAME = "PathogenKG-dataset-compgcn-ablation"
ENTITY = "giovannimaria-defilippis-university-of-naples-federico-ii"

BASE_TSV_PATH = "dataset/PathogenKG_n34_core.tsv.zip"
MODELS_PARAMS_PATH = "src/models_params.json"
MODELS_PARAMS_SWEEP_KEY = "pathogen32-cmp-gene"
MODELS_PARAMS_MODEL_KEY = "compgcn"

DATASET_VARIANTS = [
    "full_dataset",
    "noCOG",
    "noGO",
    "noCOG_GO",
    "noGO_BP",
    "noGO_CC",
    "noGO_MF",
]

GO_RELATIONS = {"BiologicalProcess", "CellularComponent", "MolecularFunction"}
BASE_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
negative_sampler = negative_sampling_filtered

SWEEP_CONFIG = {
    "name": "compgcn-dataset-variants",
    "method": "grid",
    "metric": {
        "name": "final_mixed_metric",
        "goal": "maximize",
    },
    "parameters": {
        "dataset_variant": {"values": DATASET_VARIANTS},
    },
}


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_fixed_compgcn_params():
    with open(MODELS_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params[MODELS_PARAMS_SWEEP_KEY][MODELS_PARAMS_MODEL_KEY]


def _load_base_triples(tsv_path: str) -> pd.DataFrame:
    if tsv_path.endswith(".zip"):
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, compression="zip")
    else:
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    cols = list(df.columns)
    if len(cols) < 3:
        raise ValueError(f"Dataset must have at least 3 columns, found {len(cols)}")

    rename_map = {cols[0]: "head", cols[1]: "interaction", cols[2]: "tail"}
    df = df.rename(columns=rename_map)
    return df[["head", "interaction", "tail"]].copy()


def _filter_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "full_dataset":
        return df
    if variant == "noCOG":
        return df[df["interaction"] != "ORTHOLOGY"]
    if variant == "noGO":
        return df[~df["interaction"].isin(GO_RELATIONS)]
    if variant == "noCOG_GO":
        return df[(df["interaction"] != "ORTHOLOGY") & (~df["interaction"].isin(GO_RELATIONS))]
    if variant == "noGO_BP":
        return df[df["interaction"] != "BiologicalProcess"]
    if variant == "noGO_CC":
        return df[df["interaction"] != "CellularComponent"]
    if variant == "noGO_MF":
        return df[df["interaction"] != "MolecularFunction"]
    raise ValueError(f"Unknown dataset_variant: {variant}")


def build_variant_dataset_file(tsv_path: str, variant: str) -> str:
    base_df = _load_base_triples(tsv_path)
    variant_df = _filter_variant(base_df, variant)
    if variant_df.empty:
        raise ValueError(f"Filtered dataset for variant '{variant}' is empty.")

    out_dir = Path(tempfile.gettempdir()) / "pathogenkg_dataset_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(tsv_path).stem}_{variant}.tsv"
    variant_df.to_csv(out_path, sep="\t", index=False)
    return str(out_path)


def create_compgcn_from_params(params, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
    conv_hidden_channels = {
        f"layer_{i}": params[f"layer_{i}"]
        for i in range(params["conv_layer_num"])
    }
    return compgcn(
        in_channels_dict,
        mlp_out_emb_size=params["mlp_out_layer"],
        conv_hidden_channels=conv_hidden_channels,
        num_nodes_per_type=num_nodes_per_type,
        num_entities=num_entities,
        num_relations=num_relations,
        dropout=params["dropout"],
        conv_num_layers=params["conv_layer_num"],
        opn=params["opn"],
    )

import random
def train_model():
    wandb.init()
    config = wandb.config
    dataset_variant = config.dataset_variant
    random_alphanum_quadruplet = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=4))
    wandb.run.name = f"{dataset_variant}-run-{random_alphanum_quadruplet}"

    fixed_params = load_fixed_compgcn_params()

    # Fixed training setup
    tsv_path = BASE_TSV_PATH
    task = "TARGET"
    validation_size = 0.1
    test_size = 0.2
    epochs = 200
    patience = 50
    evaluate_every = 5
    negative_rate = 1
    oversample_rate = 5
    undersample_rate = 0.5
    alpha = 0.25
    gamma = 3.0
    alpha_adv = 2.0

    seed = BASE_SEED
    set_seed(seed)

    try:
        variant_tsv_path = build_variant_dataset_file(tsv_path, dataset_variant)

        (
            in_channels_dict,
            num_nodes_per_type,
            num_entities,
            num_relations,
            train_triplets,
            train_index,
            flattened_features_per_type,
            val_triplets,
            train_val_triplets,
            test_triplets,
            train_val_test_triplets,
            _,
            _,
            _,
        ) = get_dataset(
            variant_tsv_path,
            task,
            validation_size,
            test_size,
            True,
            seed,
            oversample_rate=oversample_rate,
            undersample_rate=undersample_rate,
        )

        model = create_compgcn_from_params(
            fixed_params, in_channels_dict, num_nodes_per_type, num_entities, num_relations
        )
        model = model.to(device)
        train_index = train_index.to(device)
        flattened_features_per_type = {
            node_type: (features.to(device) if features is not None else None)
            for node_type, features in flattened_features_per_type.items()
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_params["learning_rate"])

        best_mixed_metric = -float("inf")
        patience_trigger = 0
        last_val_metric = -float("inf")

        for epoch in range(1, epochs + 1):
            training_triplets, train_labels = negative_sampler(train_triplets, int(negative_rate))
            training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)

            train_metrics = train(
                model,
                optimizer,
                fixed_params["grad_norm"],
                fixed_params["regularization"],
                flattened_features_per_type,
                train_index,
                training_triplets,
                train_labels,
                alpha,
                gamma,
                alpha_adv,
                None,
            )

            if epoch % evaluate_every == 0:
                validation_triplets, val_labels = negative_sampler(val_triplets, int(negative_rate))
                validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)

                val_metrics = test(
                    model,
                    fixed_params["regularization"],
                    flattened_features_per_type,
                    train_index,
                    validation_triplets,
                    val_labels,
                    train_val_triplets,
                    alpha,
                    gamma,
                    alpha_adv,
                    None,
                )

                mixed_metric = (
                    0.2 * val_metrics["Auroc"]
                    + 0.4 * val_metrics["Auprc"]
                    + 0.4 * val_metrics["MRR"]
                )
                last_val_metric = mixed_metric

                wandb.log(
                    {
                        "epoch": epoch,
                        "dataset_variant": dataset_variant,
                        "train_loss": train_metrics["Loss"],
                        "train_auroc": train_metrics["Auroc"],
                        "train_auprc": train_metrics["Auprc"],
                        "val_loss": val_metrics["Loss"],
                        "val_auroc": val_metrics["Auroc"],
                        "val_auprc": val_metrics["Auprc"],
                        "val_mrr": val_metrics["MRR"],
                        "val_mixed_metric": mixed_metric,
                    }
                )

                if mixed_metric > best_mixed_metric:
                    best_mixed_metric = mixed_metric
                    patience_trigger = 0
                else:
                    patience_trigger += 1
                    if patience_trigger >= patience:
                        break

        testing_triplets, test_labels = negative_sampler(test_triplets, int(negative_rate))
        testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)

        test_metrics = test(
            model,
            fixed_params["regularization"],
            flattened_features_per_type,
            train_index,
            testing_triplets,
            test_labels,
            train_val_test_triplets,
            alpha,
            gamma,
            alpha_adv,
            None,
        )

        final_mixed_metric = (
            0.2 * test_metrics["Auroc"]
            + 0.4 * test_metrics["Auprc"]
            + 0.4 * test_metrics["MRR"]
        )

        wandb.log(
            {
                "dataset_variant": dataset_variant,
                "params_group_train": "pathogen32-cmp-gene",
                "params_group_val": "pathogen32-cmp-gene-val",
                "params_group_test": "pathogen32-cmp-gene-test",
                "last_val_mixed_metric": last_val_metric,
                "best_val_mixed_metric": best_mixed_metric,
                "test_auroc": test_metrics["Auroc"],
                "test_auprc": test_metrics["Auprc"],
                "test_mrr": test_metrics["MRR"],
                "test_hits@1": test_metrics["Hits@"][1],
                "test_hits@3": test_metrics["Hits@"][3],
                "test_hits@10": test_metrics["Hits@"][10],
                "final_mixed_metric": final_mixed_metric,
            }
        )

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error: {e}")
        cleanup_cuda()
        wandb.log({"error": "CUDA_OOM", "error_details": str(e), "status": "skipped"})
        return
    except Exception as e:
        print(f"Error in training: {e}")
        wandb.log({"error": "training_error", "error_details": str(e), "status": "failed"})
        cleanup_cuda()
        raise
    finally:
        cleanup_cuda()
        wandb.finish()


def run_dataset_sweep():
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT_NAME, entity=ENTITY)
    print(f"Created sweep {sweep_id}")
    print(f"Run: wandb agent {ENTITY}/{PROJECT_NAME}/{sweep_id}")

    wandb.agent(
        sweep_id,
        train_model,
        count=len(DATASET_VARIANTS),
        project=PROJECT_NAME,
        entity=ENTITY,
    )


if __name__ == "__main__":
    print("Starting dataset sweep for fixed CompGCN params")
    print(f"Device: {device}")
    print(f"Dataset variants: {DATASET_VARIANTS}")
    run_dataset_sweep()

# %%
