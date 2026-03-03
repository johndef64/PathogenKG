#%%

import wandb
import pandas as pd


# WandB configuration
PROJECT_NAME = "PathogenKG-neg-compgcn"  # Replace with your WandB project name
# PROJECT_NAME = "PathogenKG-neg-rgcn"
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
"""
# Log final test metrics
wandb.log({
    'test_auroc': test_metrics["Auroc"],
    'test_auprc': test_metrics["Auprc"],
    'test_mrr': test_metrics["MRR"],
    'test_hits@1': test_metrics["Hits@"][1],
    'test_hits@3': test_metrics["Hits@"][3],
    'test_hits@10': test_metrics["Hits@"][10],
    'final_mixed_metric': 0.2 * test_metrics["Auroc"] + 0.4 * test_metrics["Auprc"] + 0.4 * test_metrics["MRR"]
})

final_mixed_metric
AUROC (20%): misura separazione classi, robusta ma meno sensibile a imbalance.
AUPRC (40%): cruciale per link prediction (pochi edge positivi), enfatizza precision/recall su positivi.
MRR (40%): prioritizza top ranking (essenziale per raccomandazioni link).
Pesi enfatizzano metriche ranking-specifiche vs AUROC generica.
"""

best_run = runs[df['final_mixed_metric'].idxmax()]
print(f"Migliore: {best_run.name} (ID: {best_run.id})")
print(f"Config: {best_run.config}")
# print(f"Link: {best_run.url}")
best = df.loc[df['final_mixed_metric'].idxmax()]
print(f"Best run: {best['run_name']} (ID: {best['run_id']})")
print(f"AUROC: val={best['val_auroc']:.3f}, test={best['test_auroc']:.3f}")
print(f"AUPRC: val={best['val_auprc']:.3f}, test={best['test_auprc']:.3f}")
print(f"MRR: test={best['test_mrr']:.3f}")
# print(f"Train-val gap: {best['train_val_gap']:.2%}")
print(f"Link: {best['run_url']}")
best_run.summary
#%%
"""
Migliore: confused-sweep-44 (ID: rkzmktlm)
Config: {'opn': 'corr', 'dropout': 0.2334615069280862, 'layer_0': 64, 'layer_1': 16, 'layer_2': 8, 'grad_norm': 0.5946306481700605, 'num_bases': 30, 'model_name': 'compgcn', 'learning_rate': 0.0003988884413382042, 'mlp_out_layer': 16, 'conv_layer_num': 3, 'regularization': 0.000280785439553541}
Best run: confused-sweep-44 (ID: rkzmktlm)
AUROC: val=0.416, test=0.478
AUPRC: val=0.455, test=0.522
MRR: test=0.460
Link: https://wandb.ai/giovannimaria-defilippis-university-of-naples-federico-ii/PathogenKG-neg-compgcn/runs/rkzmktlm

"""


"""
Migliore: revived-sweep-115 (ID: am4ks7x4)
Config: {'opn': 'sub', 'dropout': 0.49576013566205784, 'layer_0': 8, 'layer_1': 32, 'layer_2': 64, 'grad_norm': 1.056740604005027, 'num_bases': 30, 'model_name': 'compgcn', 'learning_rate': 0.00019390873242212473, 'mlp_out_layer': 32, 'conv_layer_num': 3, 'regularization': 0.0017364783276804412}
---
Best run: revived-sweep-115 (ID: am4ks7x4)
AUROC: val=0.412, test=0.525
AUPRC: val=0.439, test=0.527
MRR: test=0.781


compgcn:{
'opn': 'sub', 
'dropout': 0.49576013566205784,
'layer_0': 8, 
'layer_1': 32, 
'layer_2': 64,
'grad_norm': 1.056740604005027,
'num_bases': 30,
'model_name': 'compgcn', 
'learning_rate': 0.00019390873242212473, 
'mlp_out_layer': 32, 
'conv_layer_num': 3, 
'regularization': 0.0017364783276804412
}


rgcn:
Migliore: legendary-sweep-187 (ID: 8hvobrtt)
Best run: legendary-sweep-187 (ID: 8hvobrtt)
AUROC: val=0.498, test=0.529
AUPRC: val=0.566, test=0.565
MRR: test=0.833
Config: 
{'layer_0': 32, 
'layer_1': 16, 
'layer_2': 8, 
'grad_norm': 2.664619057620471, 
'num_bases': 30, 
'model_name': 'rgcn',
 'learning_rate': 0.00016364376397721685, 
 'mlp_out_layer': 32, 'conv_layer_num': 3, 
 'regularization': 0.0024514218302963153}


 {'_runtime': 49, '_step': 40,  '_timestamp': 1772134444.859578, 
   '_wandb': {'runtime': 49}, 'epoch': 200, 
   'final_mixed_metric': 0.6650137789154118, 
   'test_auprc': 0.5646539330482483, 
   'test_auroc': 0.5290942823410356, 
   'test_hits@1': 0.75, 'test_hits@10': 1, 'test_hits@3': 1, 'test_mrr': 0.8333333730697632, 'train_auprc': 0.9669190049171448, 'train_auroc': 0.970001670750724, 'train_loss': 0.02603209018707275, 'val_auprc': 0.565815806388855, 'val_auroc': 0.4983563445101907, 'val_loss': 0.05378896743059158, 'val_mixed_metric': 0.3766794089891658, 'val_mrr': 0.12670454382896423}
"""




SWEEP_CONFIG_ORIGINAL = {
	'method': 'bayes',
	'metric': {
		'name': 'val_mixed_metric',
		'goal': 'maximize'
	},
	'parameters': {
		'learning_rate': {
			'distribution': 'log_uniform_values',
			'min': 1e-4,
			'max': 1e-2
		},
		'regularization': {
			'distribution': 'log_uniform_values',
			'min': 1e-6,
			'max': 1e-2
		},
		'grad_norm': {
			'distribution': 'uniform',
			'min': 0.5,
			'max': 5.0
		},
		'dropout': {
			'distribution': 'uniform',
			'min': 0.0,
			'max': 0.7
		},
		'conv_layer_num': {
			'values': [1, 2, 3]
		},
		'mlp_out_layer': {
			'values': [8, 16, 32, 64]
		},
		'layer_0': {
			'values': [8, 16, 32, 64]
		},
		'layer_1': {
			'values': [8, 16, 32, 64]
		},
		'layer_2': {
			'values': [8, 16, 32, 64]
		},
		'num_bases': {
			'values': [10, 20, 30, 40, 50]
		},
		'opn': {  # For CompGCN only
			'values': ['mult', 'sub', 'corr']
		},

	}
}


def build_narrowed_sweep_config(top_n: int = 10) -> dict:
    """Restringe lo spazio di ricerca dello sweep usando i top_n run per ogni modello.

    Per ogni modello (compgcn / rgcn) vengono recuperati i top_n run ordinati per
    `final_mixed_metric`.  Poi:
    - parametri *continui* (log-uniform): nuovo [min, max] = [min(top_n), max(top_n)]
      con margine del 15% in log-space, CLIPPATO ai bound originali di SWEEP_CONFIG_ORIGINAL.
    - parametri *continui* (uniform): nuovo [min, max] con margine lineare del 15%,
      CLIPPATO ai bound originali di SWEEP_CONFIG_ORIGINAL.
    - parametri *discreti*: solo il/i valore/i più frequente/i nei top_n (moda),
      con fallback all'unione se la moda copre meno di 2 valori e il pool ne ha di più.
    - `opn` (solo CompGCN): moda nei top_n del progetto compgcn.
    """
    import math
    from collections import Counter

    PROJECTS = {
        'compgcn': "PathogenKG-neg-compgcn",
        'rgcn':    "PathogenKG-neg-rgcn",
    }
    LOG_UNIFORM_PARAMS = ['learning_rate', 'regularization']
    UNIFORM_PARAMS     = ['grad_norm', 'dropout']
    DISCRETE_PARAMS    = ['conv_layer_num', 'mlp_out_layer', 'layer_0', 'layer_1', 'layer_2', 'num_bases']
    COMPGCN_ONLY       = ['opn']

    # Bounds from SWEEP_CONFIG_ORIGINAL — used as hard clips
    ORIGINAL_BOUNDS = {
        'learning_rate':  (1e-4, 1e-2),
        'regularization': (1e-6, 1e-2),
        'grad_norm':      (0.5,  5.0),
        'dropout':        (0.0,  0.7),
    }

    # --- pull top-N runs for each model ---
    model_tops: dict[str, pd.DataFrame] = {}
    for model, project in PROJECTS.items():
        project_runs = api.runs(ENTITY + "/" + project)
        rows_m = []
        for r in project_runs:
            summary = dict(getattr(r.summary, "_json_dict", {}) or {})
            cfg     = {k: v for k, v in (r.config or {}).items() if not str(k).startswith("_")}
            rows_m.append({"run_id": r.id, "final_mixed_metric": summary.get("final_mixed_metric"), **cfg})

        top_df = (
            pd.DataFrame(rows_m)
            .dropna(subset=["final_mixed_metric"])
            .sort_values("final_mixed_metric", ascending=False)
            .head(top_n)
        )
        model_tops[model] = top_df
        print(
            f"[{model}] top-{top_n} metric range: "
            f"{top_df['final_mixed_metric'].min():.4f} - {top_df['final_mixed_metric'].max():.4f}"
        )

    # --- helpers ---
    def all_values(col: str, models=None) -> list:
        vals = []
        targets = {m: t for m, t in model_tops.items() if models is None or m in models}
        for top_df in targets.values():
            if col in top_df.columns:
                vals.extend(top_df[col].dropna().tolist())
        return vals

    def log_range_clipped(col: str, margin_frac: float = 0.15):
        vals = all_values(col)
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        if lo <= 0 or hi <= 0:
            return None
        # add margin in log-space
        factor = (hi / lo) ** margin_frac if hi > lo else 1.0
        lo_new = lo / max(factor, 1.01)
        hi_new = hi * max(factor, 1.01)
        # clip to original bounds
        orig_lo, orig_hi = ORIGINAL_BOUNDS[col]
        return max(lo_new, orig_lo), min(hi_new, orig_hi)

    def linear_range_clipped(col: str, margin_frac: float = 0.15):
        vals = all_values(col)
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        span = hi - lo
        margin = max(span * margin_frac, 1e-3)
        lo_new = lo - margin
        hi_new = hi + margin
        # clip to original bounds
        orig_lo, orig_hi = ORIGINAL_BOUNDS[col]
        return max(lo_new, orig_lo), min(hi_new, orig_hi)

    def discrete_mode(col: str, models=None, top_k: int = 3) -> list:
        """Return the top_k most frequent values among the top-N runs.
        Normalises float-ints (32.0 -> 32). Falls back to full union if pool is tiny."""
        raw_vals = all_values(col, models=models)
        if not raw_vals:
            return []
        # normalise
        normed = [int(v) if isinstance(v, float) and v == int(v) else v for v in raw_vals]
        counts = Counter(normed)
        # keep values whose frequency is at least half the mode frequency
        max_count = counts.most_common(1)[0][1]
        threshold = max(1, max_count // 2)
        frequent = sorted(v for v, c in counts.items() if c >= threshold)
        # ensure at least 2 options so Bayes has room to explore
        if len(frequent) < 2:
            frequent = sorted(v for v, _ in counts.most_common(top_k))
        return sorted(frequent)

    # --- build config ---
    params: dict = {}

    for p in LOG_UNIFORM_PARAMS:
        result = log_range_clipped(p)
        if result:
            lo, hi = result
            print(f"  {p}: [{lo:.2e}, {hi:.2e}]  (orig: {ORIGINAL_BOUNDS[p]})")
            params[p] = {'distribution': 'log_uniform_values',
                         'min': round(lo, 10), 'max': round(hi, 8)}

    for p in UNIFORM_PARAMS:
        result = linear_range_clipped(p)
        if result:
            lo, hi = result
            print(f"  {p}: [{lo:.4f}, {hi:.4f}]  (orig: {ORIGINAL_BOUNDS[p]})")
            params[p] = {'distribution': 'uniform',
                         'min': round(lo, 4), 'max': round(hi, 4)}

    for p in DISCRETE_PARAMS:
        vals = discrete_mode(p)
        if vals:
            print(f"  {p}: {vals}")
            params[p] = {'values': vals}

    for p in COMPGCN_ONLY:
        vals = discrete_mode(p, models=['compgcn'])
        if vals:
            print(f"  {p}: {vals}")
            params[p] = {'values': vals}

    return {
        'method': 'bayes',
        'metric': {'name': 'val_mixed_metric', 'goal': 'maximize'},
        'parameters': params,
    }


SWEEP_CONFIG_NARROWED = build_narrowed_sweep_config()
SWEEP_CONFIG_NARROWED
#%%
SWEEP_CONFIG_NARROWED = {'method': 'bayes',
 'metric': {'name': 'val_mixed_metric', 'goal': 'maximize'},
 'parameters': {'learning_rate': {'distribution': 'log_uniform_values',
   'min': 0.0001,
   'max': 0.01},
  'regularization': {'distribution': 'log_uniform_values',
   'min': 1e-06,
   'max': 0.01},
  'grad_norm': {'distribution': 'uniform', 'min': 0.5, 'max': 5.0},
  'dropout': {'distribution': 'uniform', 'min': 0.0162, 'max': 0.6582},
  'conv_layer_num': {'values': [2, 3]},
  'mlp_out_layer': {'values': [8, 16, 32, 64]},
  'layer_0': {'values': [8, 16, 32]},
  'layer_1': {'values': [16, 32]},
  'layer_2': {'values': [8, 16, 32]},
  'num_bases': {'values': [30, 50]},
  'opn': {'values': ['corr', 'mult', 'sub']}}}

SWEEP_CONFIG_NARROWED











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
"""
Best run: visionary-sweep-147 (ID: 6sxptz2w)
AUROC: val=0.722, test=0.735
AUPRC: val=0.679, test=0.702
MRR: test=0.306
Train-val gap: 19.24%
"""
best
#%%
best_run.config