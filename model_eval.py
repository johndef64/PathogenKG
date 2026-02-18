# model_eval.py
"""
Script per VALUTARE modelli già addestrati.
Non esegue alcun training, carica solamente un modello salvato e ne calcola le metriche.

Esempi d'uso:
  # Valuta un modello salvato
  python model_eval.py --model_path models/target_pathogenkg_20260218/compgcn_run0.pt --model compgcn --task TARGET

  # Specifica un dataset diverso
  python model_eval.py --model_path models/myfolder/rgcn_run0.pt --model rgcn --tsv dataset/drkg/drkg_reduced.tsv --task Compound-Gene

  # Esegui solo ranking (senza test metrics)  
  python model_eval.py --model_path models/myfolder/compgcn_run0.pt --model compgcn --task TARGET --ranking_only
"""



#%%
import warnings
warnings.simplefilter(action='ignore')

import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.utils_v2 import load_data, entities2id_offset, rel2id_offset, edge_ind_to_id, entities_features_flattening,\
                      set_target_label, triple_sampling, graph_to_undirect, negative_sampling, add_self_loops, evaluation_metrics, get_edge_type

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn

BASE_SEED = 42

# Dataset path
dataset = 'PathogenKG_n19.tsv'
DEFAULT_TRAIN_TSV = os.path.join('dataset', dataset)
models_params_path = './src/models_params.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _resolve_dataset_path(tsv_path: str) -> str:
  """Risolve e valida il path del dataset."""
  resolved = os.path.normpath(tsv_path)
  if not os.path.exists(resolved):
    raise FileNotFoundError(
      f"Dataset TSV not found: {resolved}."
    )
  return resolved

# Focal loss for evaluation metrics (same as training)
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
  """Compute focal loss for binary classification with logits."""
  bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
  p_t = torch.exp(-bce)
  loss = alpha * (1 - p_t) ** gamma * bce
  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  else:
    return loss

def get_dataset_for_eval(tsv_path, task, validation_size, test_size, quiet, seed, undersample_rate):
  """
  Carica il dataset per la valutazione.
  Simile a get_dataset ma senza oversampling (non necessario per eval).
  """
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet)
  edge_index = set_target_label(edge_index, [x for x in task.split(',')])
  
  ent2id, all_nodes_per_type = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id = rel2id_offset(edge_index)
  indexed_edge_index = edge_ind_to_id(edge_index, ent2id, relation2id)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  
  indexed_edge_index["label"] = edge_index["label"].values
  
  # Separa target e non-target
  from src.utils_v2 import select_target_triplets
  non_target_triplets, target_triplets = select_target_triplets(indexed_edge_index)
  train_triplets, val_triplets, test_triplets = triple_sampling(
    target_triplets.loc[:,["head","interaction","tail"]].values,
    validation_size, test_size, quiet, seed
  )
  
  # under-sampling non-target
  if undersample_rate < 1.0:
    rnd = np.random.RandomState(seed)
    keep = int(len(non_target_triplets) * undersample_rate)
    idx = rnd.choice(len(non_target_triplets), size=keep, replace=False)
    non_target_triplets = non_target_triplets.iloc[idx]
  
  non_target_triplets = non_target_triplets.loc[:,["head","interaction","tail"]].values
  train_index = np.concatenate([non_target_triplets, train_triplets], axis=0)
  
  num_nodes_per_type = {node_type: len(nodes) for node_type, nodes in all_nodes_per_type.items()}
  num_entities = sum(num_nodes_per_type.values())
  
  train_index = graph_to_undirect(train_index, len(relation2id))
  train_index = add_self_loops(train_index, num_entities, len(relation2id))
  train_index = torch.tensor(train_index)  # Converti a tensor
  
  train_triplets = torch.tensor(train_triplets)
  val_triplets = torch.tensor(val_triplets)
  test_triplets = torch.tensor(test_triplets)
  
  train_val_triplets = torch.cat([train_triplets, val_triplets], axis=0).to(device)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], axis=0).to(device)
  
  in_channels_dict = {node_type: (features.shape[1] if features is not None else None) 
                      for node_type, features in flattened_features_per_type.items()}
  num_relations = len(relation2id)
  
  return (in_channels_dict, num_nodes_per_type, num_entities, num_relations,
          train_triplets, train_index, flattened_features_per_type, val_triplets,
          train_val_triplets, test_triplets, train_val_test_triplets, edge_index,
          ent2id, relation2id)


def get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations):
  with open(models_params_path, 'r') as f:
    models_params = json.load(f)
  
  # Use task params if available, otherwise use "default" or fall back to first available task
  if task in models_params:
    model_params = models_params[task][model_name]
  elif "default" in models_params:
    print(f"[get_model] Task '{task}' not found in params, using 'default'")
    model_params = models_params["default"][model_name]
  else:
    # Fall back to first available task (e.g., Compound-ExtGene)
    fallback_task = list(models_params.keys())[0]
    print(f"[get_model] Task '{task}' not found in params, using '{fallback_task}' params")
    model_params = models_params[fallback_task][model_name]

  conv_hidden_channels = {f'layer_{x}':model_params[f'layer_{x}']  for x in range(model_params['conv_layer_num'])} 

  if model_name == 'rgcn':
    model = rgcn(
      in_channels_dict,
      model_params['mlp_out_layer'],
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations,
      model_params['conv_layer_num'],
      model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'rgat':
    model = rgat(
      in_channels_dict,
      model_params['mlp_out_layer'],
      model_params['mlp_out_layer'],
      conv_hidden_channels,
      num_nodes_per_type,
      num_entities,
      num_relations,
      conv_num_layers=model_params['conv_layer_num'],
      num_bases=model_params['num_bases'],
      activation_function=F.relu,
      device=device
    )
  elif model_name == 'compgcn':
    model = compgcn(
      in_channels_dict,
      mlp_out_emb_size=model_params['mlp_out_layer'],
      conv_hidden_channels=conv_hidden_channels,
      num_nodes_per_type=num_nodes_per_type,
      num_entities=num_entities,
      num_relations=num_relations,
      dropout=model_params['dropout'],
      conv_num_layers=model_params['conv_layer_num'],
      opn=model_params['opn'],
    )
  else:
    return None
  
  return model, model_params['learning_rate'], model_params['regularization'], model_params['grad_norm']


def test(model, reg_param, x_dict, index, target_triplets, target_labels, train_val_triplets, alpha, gamma, alpha_adv, change_points=None):    
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}

  model.eval()
  with torch.no_grad():
    if change_points != None:
      out = model(x_dict, index, change_points)
    else:
      out = model(x_dict, index)
    scores = model.distmult(out, target_triplets)

    # loss = model.score_loss(scores, target_labels)
    # reg_loss = (reg_param * model.reg_loss(out, target_triplets))
    # loss =  loss  + reg_loss

    # Compute focal loss instead of standard BCE
    # loss_fl = focal_loss(scores, target_labels)
    # reg_loss = reg_param * model.reg_loss(out, target_triplets)
    # loss = loss_fl + reg_loss

    ############################################
    # focal loss + hard-negative mining | NEW! #
    ############################################
    raw_logits = model.distmult(out, target_triplets)
    probs = torch.sigmoid(raw_logits)
    # Per-sample focal loss (no reduction)
    fl_all = focal_loss(raw_logits, target_labels, alpha, gamma, reduction='none')
    # Separate positives and negatives
    pos_mask = target_labels.bool()
    neg_mask = ~pos_mask
    # Positive focal loss averaged
    pos_loss = fl_all[pos_mask].mean()
    # Adversarial weighting for negatives
    neg_probs = probs[neg_mask]
    neg_weights = torch.softmax(neg_probs * alpha_adv, dim=0)
    # Weighted negative focal loss
    neg_loss = (neg_weights * fl_all[neg_mask]).sum()
    # Regularization loss
    reg_loss = reg_param * model.reg_loss(out, target_triplets)
    # Total loss
    loss = pos_loss + neg_loss + reg_loss
    # Use probabilities for metrics
    scores = probs

    scores = torch.sigmoid(scores)
  
  mrr,hits = evaluation_metrics(model, out, train_val_triplets, target_triplets[target_labels.to(torch.int).view(-1)], 20, 0, hits=[1,3,10])

  metrics["Loss"]   = loss.item()
  metrics["MRR"]    = mrr
  metrics["Hits@"]  = hits
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()
  
  return metrics


# Import the helper function from utils_v2
from src.utils_v2 import get_edge_type

def eval(model, flattened_features_per_type, train_index, edge_index, ent2id, relation2id, change_points=None, task=None):
    """
    Rank entities by prediction confidence for novel interactions of a given task type.
    
    Args:
        model: The trained model.
        flattened_features_per_type: Node features by type.
        train_index: Training edge index.
        edge_index: Original edge index DataFrame with 'head', 'interaction', 'tail' columns.
        ent2id: Entity to ID mapping.
        relation2id: Relation to ID mapping.
        change_points: Change points for RGAT (optional).
        task: The task/edge type to evaluate (e.g., 'Compound-ExtGene'). 
              If None, uses args.task.
              
    Note:
        Edge types are derived on-the-fly from entity prefixes (e.g., 'Compound::x' -> 'Compound').
        No 'type' column is needed in the edge_index DataFrame.
    """
    model.eval()
    
    # Use provided task or fall back to args.task
    eval_task = task if task is not None else args.task
    
    def normalize(s):
        """Normalize string for matching: lowercase, replace spaces with underscores"""
        return str(s).lower().replace(" ", "_")
    
    # Build normalized set of targets (including reversed edge types)
    eval_task_normalized = {normalize(eval_task)}
    if '-' in eval_task:
        parts = eval_task.split('-')
        if len(parts) == 2:
            eval_task_normalized.add(normalize(f"{parts[1]}-{parts[0]}"))
    
    print(f"[i] Evaluating model for novel {eval_task} interactions...")
    with torch.no_grad():
        # Get embeddings from the model
        if change_points != None:
            out = model(flattened_features_per_type, train_index, change_points)
        else:
            out = model(flattened_features_per_type, train_index)
        
        # Filter edges by task - match either edge type OR interaction name
        task_mask = edge_index.apply(
            lambda row: (
                normalize(get_edge_type(row['head'], row['tail'])) in eval_task_normalized or
                normalize(row['interaction']) in eval_task_normalized
            ),
            axis=1
        )
        task_edges = edge_index[task_mask]
        
        if len(task_edges) == 0:
            # Show available options for debugging
            available_types = edge_index.apply(
                lambda row: get_edge_type(row['head'], row['tail']), axis=1
            ).unique().tolist()
            available_interactions = edge_index['interaction'].unique().tolist()
            print(f"[WARNING] No edges found for task '{eval_task}'.")
            print(f"  Available edge types: {available_types}")
            print(f"  Available interactions: {available_interactions}")
            return []
        
        # Get unique head and tail entities for this task
        heads = set(task_edges['head'].unique())
        tails = set(task_edges['tail'].unique())
        
        # Determine the relation name dynamically from the dataset
        # Use the most common interaction type for this edge type
        relation_name = task_edges['interaction'].mode().iloc[0] if len(task_edges) > 0 else "TARGET"
        relation_name = relation_name.replace(" ", "_")  # Normalize whitespace
        
        if relation_name not in relation2id:
            print(f"[WARNING] Relation '{relation_name}' not found in relation2id. Available: {list(relation2id.keys())}")
            # Try to find a matching relation
            possible_relations = [r for r in relation2id.keys() if not r.startswith('rev_')]
            if possible_relations:
                relation_name = possible_relations[0]
                print(f"[i] Using fallback relation: {relation_name}")
            else:
                return []
        
        rel_id = relation2id[relation_name]
        
        # Generate all possible triplets for this task
        all_triplets = []
        for head_entity in heads:
            for tail_entity in tails:
                if head_entity in ent2id and tail_entity in ent2id:
                    head_id = ent2id[head_entity]
                    tail_id = ent2id[tail_entity]
                    all_triplets.append([head_id, rel_id, tail_id])
        
        if not all_triplets:
            return []
        
        all_triplets = torch.tensor(all_triplets).to(device)
        
        # Get prediction scores
        scores = model.distmult(out, all_triplets)
        scores = torch.sigmoid(scores)
        
        # Create ranked list of predictions
        ranked_predictions = []
        id2ent = {v: k for k, v in ent2id.items()}
        
        for score, triplet in zip(scores, all_triplets):
            head_id = triplet[0].item()
            tail_id = triplet[2].item()
            
            head = id2ent.get(head_id, f"unknown_head_{head_id}")
            tail = id2ent.get(tail_id, f"unknown_tail_{tail_id}")
            
            ranked_predictions.append({
                'head': head,
                'tail': tail,
                'confidence': score.item(),
                'relation': eval_task
            })
        
        # Sort by confidence (descending)
        ranked_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"[i] Total predictions made: {len(ranked_predictions)}")
        
        return ranked_predictions

def main_eval(model_path, model_name, dataset_tsv, task, validation_size, test_size, quiet,
              negative_rate, undersample_rate, alpha, gamma, alpha_adv, ranking_only):
  """
  Funzione principale per valutare un modello già addestrato.
  
  Args:
    model_path: Path al file .pt del modello salvato
    model_name: Tipo di modello (rgcn, rgat, compgcn)
    dataset_tsv: Path al dataset TSV
    task: Task da valutare
    validation_size: Percentuale validazione
    test_size: Percentuale test
    quiet: Stampa meno output
    negative_rate: Rate di negative sampling
    undersample_rate: Frazione di non-target triplets da mantenere
    alpha, gamma, alpha_adv: Parametri focal loss
    ranking_only: Se True, genera solo il ranking senza metriche sul test set
  """
  
  print(f'[i] Caricamento modello da: {model_path}')
  
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modello non trovato: {model_path}")
  
  seed = BASE_SEED
  
  # Carica dataset
  print('[i] Caricamento dataset...', end='', flush=True)
  start_time = time.time()
  
  in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
  train_triplets, train_index, flattened_features_per_type, val_triplets, \
  train_val_triplets, test_triplets, train_val_test_triplets, \
  edge_index, ent2id, relation2id = get_dataset_for_eval(
    dataset_tsv, task, validation_size, test_size, quiet, seed, undersample_rate
  )
  
  print(f' ok ({time.time() - start_time:.2f}s)')
  
  # Crea modello (con la stessa architettura)
  model, lr, regularization, grad_norm = get_model(
    model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations
  )
  
  # Carica i pesi salvati
  model.load_state_dict(torch.load(model_path, map_location=device))
  model = model.to(device)
  model.eval()
  
  # Move data to device
  train_index = train_index.to(device)
  flattened_features_per_type = {
    node_type: (features.to(device) if features is not None else None)
    for node_type, features in flattened_features_per_type.items()
  }
  
  # Calcola change_points per RGAT
  change_points = None
  if model_name == 'rgat':
    rel_ids_sorted = train_index[:, 1]
    change_points = torch.cat([
      torch.tensor([0], device=device),
      (rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
      torch.tensor([rel_ids_sorted.size(0)], device=device)
    ])
  
  metrics = None
  
  # Valutazione sul test set (se richiesto)
  if not ranking_only:
    print('[i] Valutazione sul test set...')
    with torch.no_grad():
      testing_triplets, test_labels = negative_sampling(test_triplets, negative_rate)
      testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
      
      metrics = test(
        model, regularization, flattened_features_per_type, train_index,
        testing_triplets, test_labels, train_val_test_triplets,
        alpha, gamma, alpha_adv, change_points
      )
    
    print(f"\n{'='*60}")
    print(f"RISULTATI TEST:")
    print(f"  AUROC:   {metrics['Auroc']:.4f}")
    print(f"  AUPRC:   {metrics['Auprc']:.4f}")
    print(f"  MRR:     {metrics['MRR']:.4f}")
    print(f"  Hits@1:  {metrics['Hits@'][1]:.4f}")
    print(f"  Hits@3:  {metrics['Hits@'][3]:.4f}")
    print(f"  Hits@10: {metrics['Hits@'][10]:.4f}")
    print(f"  Loss:    {metrics['Loss']:.4f}")
    print(f"{'='*60}\n")
  
  # Genera ranking delle predizioni
  print('[i] Generazione ranking predizioni...')
  rank = eval(model, flattened_features_per_type, train_index, edge_index, 
              ent2id, relation2id, change_points, task=task)
  
  # Mostra top predictions
  top_p_num = min(10, len(rank))
  print(f"\n[i] Top {top_p_num} predizioni:")
  print("-" * 80)
  for idx, pred in enumerate(rank[:top_p_num], 1):
    print(f"  {idx:2d}. {pred['head'][:35]:35s} -> {pred['tail'][:25]:25s} | {pred['confidence']:.4f}")
  print("-" * 80)
  
  # Salva risultati
  output_dir = os.path.dirname(model_path)
  base_name = os.path.splitext(os.path.basename(model_path))[0]
  
  # Salva ranking
  ranking_path = os.path.join(output_dir, f"{base_name}_eval_ranking.json")
  with open(ranking_path, 'w') as f:
    json.dump(rank, f, indent=2)
  print(f"[i] Ranking salvato in: {ranking_path}")
  
  # Salva metriche
  if metrics is not None:
    metrics_save = {
      "Auroc": metrics["Auroc"],
      "Auprc": metrics["Auprc"],
      "MRR": metrics["MRR"],
      "Hits@1": metrics["Hits@"][1],
      "Hits@3": metrics["Hits@"][3],
      "Hits@10": metrics["Hits@"][10],
      "Loss": metrics["Loss"]
    }
    metrics_path = os.path.join(output_dir, f"{base_name}_eval_metrics.json")
    with open(metrics_path, 'w') as f:
      json.dump(metrics_save, f, indent=2)
    print(f"[i] Metriche salvate in: {metrics_path}")
  
  return metrics, rank


if __name__ == '__main__':
  print(f'[i] Model Evaluation Script')
  print(f'[i] Running on {device}')
  
  parser = argparse.ArgumentParser(
    description='Valuta un modello GNN già addestrato.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Esempi:
  # Valuta un modello CompGCN
  python model_eval.py --model_path models/target_pathogenkg/compgcn_run0.pt --model compgcn --task TARGET

  # Valuta specificando un dataset diverso
  python model_eval.py --model_path models/myfolder/rgcn.pt --model rgcn --tsv dataset/drkg/drkg_reduced.tsv --task Compound-Gene

  # Solo ranking (senza metriche test)
  python model_eval.py --model_path models/myfolder/compgcn.pt --model compgcn --task TARGET --ranking_only
    """
  )
  
  # Argomenti principali
  parser.add_argument('--model_path', type=str, required=True,
                      help='Path al file .pt del modello da valutare')
  parser.add_argument('-m', '--model', type=str, default='compgcn', 
                      choices=['rgcn', 'rgat', 'compgcn'],
                      help='Tipo di modello (default: compgcn)')
  parser.add_argument('--tsv', type=str, default=DEFAULT_TRAIN_TSV,
                      help=f'Path al dataset TSV (default: {DEFAULT_TRAIN_TSV})')
  parser.add_argument('--task', type=str, default='TARGET',
                      help='Task da valutare (default: TARGET)')
  
  # Parametri dataset
  parser.add_argument('--validation_size', type=float, default=0.1,
                      help='Dimensione validation set (default: 0.1)')
  parser.add_argument('--test_size', type=float, default=0.2,
                      help='Dimensione test set (default: 0.2)')
  parser.add_argument('--undersample_rate', type=float, default=0.5,
                      help='Frazione non-target triplets da mantenere (default: 0.5)')
  parser.add_argument('--negative_rate', type=float, default=1,
                      help='Negative sampling rate (default: 1)')
  
  # Parametri focal loss
  parser.add_argument('--alpha', type=float, default=0.25,
                      help='Alpha focal loss (default: 0.25)')
  parser.add_argument('--gamma', type=float, default=3.0,
                      help='Gamma focal loss (default: 3.0)')
  parser.add_argument('--alpha_adv', type=float, default=2.0,
                      help='Alpha hard-negative mining (default: 2.0)')
  
  # Opzioni
  parser.add_argument('--quiet', action='store_true',
                      help='Output ridotto')
  parser.add_argument('--ranking_only', action='store_true',
                      help='Genera solo ranking, senza metriche sul test set')
  
  args = parser.parse_args()
  
  # Risolvi path dataset
  dataset_tsv = _resolve_dataset_path(args.tsv)
  
  print(f'[i] Dataset: {os.path.basename(dataset_tsv)}')
  print(f'[i] Modello: {args.model}')
  print(f'[i] Task: {args.task}')
  print()
  
  # Esegui valutazione
  main_eval(
    model_path=args.model_path,
    model_name=args.model,
    dataset_tsv=dataset_tsv,
    task=args.task,
    validation_size=args.validation_size,
    test_size=args.test_size,
    quiet=args.quiet,
    negative_rate=args.negative_rate,
    undersample_rate=args.undersample_rate,
    alpha=args.alpha,
    gamma=args.gamma,
    alpha_adv=args.alpha_adv,
    ranking_only=args.ranking_only
  )