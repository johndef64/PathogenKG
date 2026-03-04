#%%
"""
training.py — WandB-logged training on full datasets (PathogenKG / DRKG).

Loads model hyperparameters from models_params.json (fixed per sweep key),
varies only the seed across runs so you can evaluate model stability on WandB.

Usage examples:
  # PathogenKG (default)
  python training.py --task TARGET --runs 5 --epochs 400

  # DRKG reduced
  python training.py --tsv dataset/drkg/drkg_reduced.zip --task CMP_BIND --runs 5 --epochs 300 --sweep_key pathogen32-cmp-gene-neg-fix

  # DRKG full with TREATMENT task
  python training.py --tsv dataset/drkg/drkg.tsv --task TREATMENT --runs 6 --epochs 200

  # Change model
  python training.py --model rgcn --runs 3

  # Custom project name
  python training.py --wandb_project MyProject --runs 5
"""

import gc
import copy
import json
import time
import argparse
import numpy as np
import torch
import wandb
import random
import os

from src.utils import set_seed
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn
from src.hetero_rgcn import HeterogeneousRGCN as rgcn

from train_and_eval import (
    get_dataset,
    train,
    test,
    focal_loss,
    negative_sampling_filtered,
    negative_sampling,
    eval as eval_ranking,
)

MODELS_PARAMS_PATH = "src/models_params.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FILTERED_EVAL = True   # True = filtered (standard KGE), False = legacy

def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_model_params(sweep_key: str, model_name: str) -> dict:
    with open(MODELS_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)
    if sweep_key not in params:
        raise KeyError(f"Sweep key '{sweep_key}' not found in {MODELS_PARAMS_PATH}. Available: {list(params.keys())}")
    if model_name not in params[sweep_key]:
        raise KeyError(f"Model '{model_name}' not found under sweep '{sweep_key}'. Available: {list(params[sweep_key].keys())}")
    return params[sweep_key][model_name]


def create_model(model_name, params, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
    import torch.nn.functional as F

    conv_hidden_channels = {
        f"layer_{i}": params[f"layer_{i}"]
        for i in range(params["conv_layer_num"])
    }

    if model_name == "compgcn":
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
    elif model_name == "rgcn":
        return rgcn(
            in_channels_dict,
            params["mlp_out_layer"],
            params["mlp_out_layer"],
            conv_hidden_channels,
            num_nodes_per_type,
            num_entities,
            num_relations,
            params["conv_layer_num"],
            params["num_bases"],
            activation_function=F.relu,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'compgcn' or 'rgcn'.")


def train_single_run(args, run_index: int, seed: int):
    """Execute one training run with a given seed, logging everything to WandB."""

    rng = random.Random()
    tag = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=4))

    dataset_name = os.path.splitext(os.path.basename(args.tsv))[0]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "model": args.model,
            "task": args.task,
            "dataset": dataset_name,
            "seed": seed,
            "run_index": run_index,
            "epochs": args.epochs,
            "patience": args.patience,
            "evaluate_every": args.evaluate_every,
            "negative_rate": args.negative_rate,
            "oversample_rate": args.oversample_rate,
            "undersample_rate": args.undersample_rate,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "alpha_adv": args.alpha_adv,
            "sweep_key": args.sweep_key,
            "validation_size": args.validation_size,
            "test_size": args.test_size,
        },
        reinit=True,
    )
    wandb.run.name = f"{dataset_name}-{args.task}-seed{seed}-{tag}"

    fixed_params = load_model_params(args.sweep_key, args.model)
    wandb.config.update({"model_params": fixed_params}, allow_val_change=True)

    set_seed(seed)

    neg_sampler = negative_sampling_filtered if args.negative_sampling == "filtered" else negative_sampling
    use_filtered = args.negative_sampling == "filtered"

    try:
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
            edge_index,
            ent2id,
            relation2id,
        ) = get_dataset(
            args.tsv,
            args.task,
            args.validation_size,
            args.test_size,
            True,  # quiet
            seed,
            oversample_rate=args.oversample_rate,
            undersample_rate=args.undersample_rate,
        )

        all_entities_arr = np.arange(num_entities)
        all_true_arr = train_val_test_triplets.cpu().numpy()

        model = create_model(
            args.model, fixed_params, in_channels_dict, num_nodes_per_type, num_entities, num_relations
        )
        model = model.to(device)
        train_index_dev = train_index.to(device)
        feat_dev = {
            nt: (f.to(device) if f is not None else None)
            for nt, f in flattened_features_per_type.items()
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=fixed_params["learning_rate"])

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        best_mixed_metric = -float("inf")

        for epoch in range(1, args.epochs + 1):
            # Negative sampling
            if use_filtered:
                training_triplets, train_labels = neg_sampler(
                    train_triplets, all_entities_arr, int(args.negative_rate), all_true_arr, seed=seed + epoch
                )
            else:
                training_triplets, train_labels = neg_sampler(train_triplets, int(args.negative_rate))
            training_triplets = training_triplets.to(device)
            train_labels = train_labels.to(device)

            train_metrics = train(
                model, optimizer,
                fixed_params["grad_norm"],
                fixed_params["regularization"],
                feat_dev, train_index_dev,
                training_triplets, train_labels,
                args.alpha, args.gamma, args.alpha_adv,
                None,
            )

            # Validation
            if epoch % args.evaluate_every == 0:
                if use_filtered:
                    validation_triplets, val_labels = neg_sampler(
                        val_triplets, all_entities_arr, int(args.negative_rate), all_true_arr, seed=seed + 10_000
                    )
                else:
                    validation_triplets, val_labels = neg_sampler(val_triplets, int(args.negative_rate))
                validation_triplets = validation_triplets.to(device)
                val_labels = val_labels.to(device)

                val_metrics = test(
                    model,
                    fixed_params["regularization"],
                    feat_dev, train_index_dev,
                    validation_triplets, val_labels,
                    train_val_triplets,
                    args.alpha, args.gamma, args.alpha_adv,
                    None,
                    use_filtered_eval=USE_FILTERED_EVAL,
                    all_target_triplets=train_val_test_triplets,
                    num_entities=num_entities,
                )

                mixed_metric = (
                    0.2 * val_metrics["Auroc"]
                    + 0.4 * val_metrics["Auprc"]
                    + 0.4 * val_metrics["MRR"]
                )

                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["Loss"],
                    "train_auroc": train_metrics["Auroc"],
                    "train_auprc": train_metrics["Auprc"],
                    "val_loss": val_metrics["Loss"],
                    "val_auroc": val_metrics["Auroc"],
                    "val_auprc": val_metrics["Auprc"],
                    "val_mrr": val_metrics["MRR"],
                    "val_hits@1": val_metrics["Hits@"][1],
                    "val_hits@3": val_metrics["Hits@"][3],
                    "val_hits@10": val_metrics["Hits@"][10],
                    "val_mixed_metric": mixed_metric,
                })

                # Early stopping on val loss
                val_loss = val_metrics["Loss"]
                if val_loss < (best_val_loss - args.min_delta):
                    best_val_loss = val_loss
                    best_mixed_metric = mixed_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if args.early_stopping and patience_counter >= args.patience:
                        print(f"[i] Early stopping at epoch {epoch} (patience={args.patience})")
                        break
            else:
                # Log train metrics every epoch
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_metrics["Loss"],
                    "train_auroc": train_metrics["Auroc"],
                    "train_auprc": train_metrics["Auprc"],
                })

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final test
        if use_filtered:
            testing_triplets, test_labels = neg_sampler(
                test_triplets, all_entities_arr, int(args.negative_rate), all_true_arr, seed=seed + 20_000
            )
        else:
            testing_triplets, test_labels = neg_sampler(test_triplets, int(args.negative_rate))
        testing_triplets = testing_triplets.to(device)
        test_labels = test_labels.to(device)

        test_metrics = test(
            model,
            fixed_params["regularization"],
            feat_dev, train_index_dev,
            testing_triplets, test_labels,
            train_val_test_triplets,
            args.alpha, args.gamma, args.alpha_adv,
            None,
            use_filtered_eval=USE_FILTERED_EVAL,
            all_target_triplets=train_val_test_triplets,
            num_entities=num_entities,
        )

        final_mixed = (
            0.2 * test_metrics["Auroc"]
            + 0.4 * test_metrics["Auprc"]
            + 0.4 * test_metrics["MRR"]
        )

        wandb.log({
            "seed": seed,
            "best_val_loss": best_val_loss,
            "best_val_mixed_metric": best_mixed_metric,
            "test_auroc": test_metrics["Auroc"],
            "test_auprc": test_metrics["Auprc"],
            "test_mrr": test_metrics["MRR"],
            "test_hits@1": test_metrics["Hits@"][1],
            "test_hits@3": test_metrics["Hits@"][3],
            "test_hits@10": test_metrics["Hits@"][10],
            "test_loss": test_metrics["Loss"],
            "final_mixed_metric": final_mixed,
        })

        print(
            f"Run {run_index} (seed={seed}) | "
            f"Test AUROC={test_metrics['Auroc']:.4f}  AUPRC={test_metrics['Auprc']:.4f}  "
            f"MRR={test_metrics['MRR']:.4f}  Mixed={final_mixed:.4f}"
        )

        # Save model
        if not args.no_save:
            save_dir = os.path.join("models", f"training_{dataset_name}_{args.task}_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.model}_seed{seed}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[i] Model saved to {save_path}")

            # Ranking
            rank = eval_ranking(model, feat_dev, train_index_dev, edge_index, ent2id, relation2id, None, task=args.task)
            if rank:
                ranking_path = save_path.replace(".pt", "_ranking.json")
                with open(ranking_path, "w") as f:
                    json.dump(rank[:100], f, indent=2)  # top 100
                print(f"[i] Ranking saved to {ranking_path}")

        return test_metrics

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}")
        cleanup_cuda()
        wandb.log({"error": "CUDA_OOM", "error_details": str(e)})
        return None
    except Exception as e:
        print(f"Error: {e}")
        wandb.log({"error": "training_error", "error_details": str(e)})
        cleanup_cuda()
        raise
    finally:
        cleanup_cuda()
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="WandB-logged training on full dataset with seed variation.")

    # Dataset & task
    parser.add_argument("--tsv", type=str, default="dataset/PathogenKG_n34_core.tsv.zip",
                        help="Path to training TSV/ZIP file.")
    parser.add_argument("--task", type=str, default="TARGET",
                        help="Target relation(s), comma-separated (e.g. 'TARGET' or 'CMP_BIND,ENZYME').")
    parser.add_argument("--model", type=str, default="compgcn", choices=["rgcn", "compgcn"],
                        help="Model architecture.")
    parser.add_argument("--sweep_key", type=str, default="pathogen32-cmp-gene-neg-fix",
                        help="Key in models_params.json for hyperparameters.")

    # Training
    parser.add_argument("--runs", type=int, default=5, help="Number of runs (each with a different seed).")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed; run i uses base_seed + i.")
    parser.add_argument("--random_seeds", action="store_true",
                        help="Use fully random seeds instead of sequential base_seed + i.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--evaluate_every", type=int, default=5)
    parser.add_argument("--negative_sampling", type=str, default="filtered", choices=["filtered", "standard"])
    parser.add_argument("--negative_rate", type=float, default=1)
    parser.add_argument("--oversample_rate", type=int, default=5)
    parser.add_argument("--undersample_rate", type=float, default=0.5)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)

    # Focal loss
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--alpha_adv", type=float, default=2.0)

    # WandB
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project name (default: auto-generated from dataset+model).")
    parser.add_argument("--wandb_entity", type=str,
                        default="giovannimaria-defilippis-university-of-naples-federico-ii")

    # Misc
    parser.add_argument("--no_save", action="store_true", help="Skip saving models and rankings.")

    args = parser.parse_args()

    # Auto-generate project name if not given
    if args.wandb_project is None:
        dataset_name = os.path.splitext(os.path.basename(args.tsv))[0]
        dataset_name = dataset_name.split(".")[0].split("_")[0]  # e.g. "PathogenKG" from "PathogenKG_n34_core"
        task_clean = args.task.lower().replace(",", "-")
        args.wandb_project = f"{dataset_name}-{args.model}-{task_clean}-training"

    # Resolve dataset path
    if not os.path.exists(args.tsv):
        raise FileNotFoundError(f"Dataset not found: {args.tsv}")

    print(f"[i] Device: {device}")
    print(f"[i] Dataset: {args.tsv}")
    print(f"[i] Task: {args.task}")
    print(f"[i] Model: {args.model} (params from '{args.sweep_key}')")
    print(f"[i] Runs: {args.runs}, Epochs: {args.epochs}")
    print(f"[i] WandB project: {args.wandb_project}")

    # Generate seeds
    if args.random_seeds:
        rng = random.Random()
        seeds = [rng.randint(0, 1_000_000) for _ in range(args.runs)]
    else:
        seeds = [args.base_seed + i for i in range(args.runs)]

    print(f"[i] Seeds: {seeds}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{args.runs} — seed {seed}")
        print(f"{'='*60}")
        result = train_single_run(args, i, seed)
        if result is not None:
            all_results.append(result)

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"Summary over {len(all_results)} runs:")
        for key in ["Auroc", "Auprc", "MRR"]:
            vals = [r[key] for r in all_results]
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        for h in [1, 3, 10]:
            vals = [r["Hits@"][h] for r in all_results]
            print(f"  Hits@{h}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
