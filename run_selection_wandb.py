#%%

import wandb
import pandas as pd


# WandB configuration
PROJECT_NAME = "PathogenKG-Hyperparameter-Optimization-rgcn"  # Replace with your WandB project name
# ENTITY = "gidek"  # Replace with your WandB entity
ENTITY = "giovannimaria-defilippis-university-of-naples-federico-ii"  

api = wandb.Api()
runs = api.runs(ENTITY + "/" + PROJECT_NAME)
rows = []
for r in runs:
    summary = dict(getattr(r.summary, "_json_dict", {}) or {})
    config = {
        f"config.{k}": v
        for k, v in (r.config or {}).items()
        if not str(k).startswith("_")
    }
    rows.append(
        {
            "run_id": r.id,
            "run_name": r.name,
            "run_url": r.url,
            **summary,
            **config,
        }
    )
df = pd.json_normalize(rows)
df
#%%
# df['custom_score'] = 0.7 * df.get('val/accuracy', 0) - 0.3 * df.get('val/loss', float('inf'))
# get best run based on final_mixed_metric

best_run = runs[df['final_mixed_metric'].idxmax()]
print(f"Migliore: {best_run.name} (ID: {best_run.id})")
print(f"Config: {best_run.config}")
print(f"Link: {best_run.url}")
print("---")
best_run.summary
#%%
"""
cerca queste run nel df

test_auroc ≈ val_auroc (>0.75 desiderabile)
test_auprc ≈ val_auprc (>0.65 desiderabile) 
test_mrr < 0.3 (realistico)
gap train-val < 20%

"""
def get_best_run():
    # Criteria:
    # test_auroc ≈ val_auroc (>0.75 desirable)
    # test_auprc ≈ val_auprc (>0.65 desirable)
    # test_mrr < 0.3 (realistic)
    # gap train-val < 20%
    
    # Ensure required columns exist
    required = ['test_auroc', 'val_auroc', 'test_auprc', 'val_auprc', 'test_mrr', 'train_auroc']
    for col in required:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return None

    # Work on a copy to avoid modifying the original df used elsewhere
    work = df.copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors='coerce')

    # Calculate gaps
    work['auroc_gap'] = (work['test_auroc'] - work['val_auroc']).abs()
    work['auprc_gap'] = (work['test_auprc'] - work['val_auprc']).abs()
    work['train_val_gap'] = (work['train_auroc'] - work['val_auroc']).abs() / work['train_auroc'].clip(lower=1e-6)

    # Filter by criteria
    filtered = work[
        (work['val_auroc'] > 0.75) &
        (work['test_auroc'] > 0.75) &
        (work['auroc_gap'] < 0.05) &
        (work['val_auprc'] > 0.65) &
        (work['test_auprc'] > 0.65) &
        (work['auprc_gap'] < 0.05) &
        (work['test_mrr'] < 0.3) &
        (work['train_val_gap'] < 0.2)
    ]

    if filtered.empty:
        # Fallback: choose the run with the lowest normalized distance from targets
        penalty = (
            (0.75 - work['val_auroc']).clip(lower=0) +
            (0.75 - work['test_auroc']).clip(lower=0) +
            (work['auroc_gap'] - 0.05).clip(lower=0) +
            (0.65 - work['val_auprc']).clip(lower=0) +
            (0.65 - work['test_auprc']).clip(lower=0) +
            (work['auprc_gap'] - 0.05).clip(lower=0) +
            (work['test_mrr'] - 0.3).clip(lower=0) +
            (work['train_val_gap'] - 0.2).clip(lower=0)
        )
        work['criteria_penalty'] = penalty.fillna(float('inf'))
        best = work.sort_values(['criteria_penalty', 'val_auroc', 'val_auprc'], ascending=[True, False, False]).iloc[0]
        print("Nessuna run soddisfa tutti i criteri: selezionata la più vicina.")
        print(f"Penalty: {best['criteria_penalty']:.4f} (più basso è meglio)")
    else:
        # Sort by val_auroc and val_auprc, then lowest test_mrr
        best = filtered.sort_values(['val_auroc', 'val_auprc', 'test_mrr'], ascending=[False, False, True]).iloc[0]

    print(f"Best run: {best['run_name']} (ID: {best['run_id']})")
    print(f"AUROC: val={best['val_auroc']:.3f}, test={best['test_auroc']:.3f}")
    print(f"AUPRC: val={best['val_auprc']:.3f}, test={best['test_auprc']:.3f}")
    print(f"MRR: test={best['test_mrr']:.3f}")
    print(f"Train-val gap: {best['train_val_gap']:.2%}")
    print(f"Link: {best['run_url']}")
    return best

best = get_best_run()
best
#%%
best_run.config