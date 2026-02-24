# train.py
################# SHHHH MAN-AT-WORK ####################
#           _
#     _____|~~\_____      _____________
# _-~               \    |    \
# _-    | )     \    |__/   \   \
# _-         )   |   |  |     \  \
# _-    | )     /    |--|      |  |
# __-_______________ /__/_______|  |_________
# (                |----         |  |
# `---------------'--\\\\      .`--'
#                             `||||

# # Specify a different target
# python train_and_eval.py --target 224308

# # Use a different model
# python train_and_eval.py --model rgcn

# # Change number of runs and epochs
# python train_and_eval.py --runs 5 --epochs 200

# # Training with pretraining
# python train_and_eval.py --pretrain_epochs 50

# # Training with focal loss parameters
# python train_and_eval.py --alpha 0.25 --gamma 3.0 --alpha_adv 2.0

########################################################

# # Full configuration with multiple parameters
# python train_and_eval.py --target 83332 --model compgcn --runs 3 --epochs 300 --patience 100 --validation_size 0.15 --test_size 0.2 --evaluate_every 10 --negative_rate 2 --oversample_rate 5 --undersample_rate 0.5

# # With pretraining and frozen base layers
# python train_and_eval.py --model rgat --pretrain_epochs 100 --freeze_base --epochs 200

# # Training on target graph alone with custom sampling
# python train_and_eval.py --alone --oversample_rate 10 --undersample_rate 0.3 --negative_rate 3

# # Quiet mode (minimal output)
# python train_and_eval.py --quiet --runs 5

"""
In this new version of train and eval and network.utils (v2),
you need to adapt the loading and use of the dataset for datasets formatted as triples:
which can be 
head, interaction, tail
but the column names do not matter because they are always triples. 
If the names are different, fix them when you load the dataset in load_data. Just specify which columns to take as head, interaction and tail, and then the rest of the code will work the same.

However, please note that this version also uses a 'type' column. I do not understand the purpose of this, and the scripts must be adapted for use with simple triple datasets without this 'type' column, and the networks must work in the same way.

Make these changes so that the networks work correctly.

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
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.utils import set_seed, load_data, select_target_triplets, entities2id_offset, rel2id_offset, edge_ind_to_id, entities_features_flattening,\
                      set_target_label, triple_sampling, graph_to_undirect, negative_sampling, negative_sampling_filtered, add_self_loops, evaluation_metrics

from src.hetero_rgcn import HeterogeneousRGCN as rgcn
from src.hetero_rgat import HeterogeneousRGAT as rgat
from src.hetero_compgcn import HeterogeneousCompGCN as compgcn

BASE_SEED = 42

# Single, pre-merged dataset ready for training.
dataset = 'PathogenKG_merged.tsv'
dataset = 'PathogenKG_n19.tsv'
dataset = 'PathogenKG_n19.tsv'
DEFAULT_TRAIN_TSV = os.path.join('dataset', dataset)
models_params_path = './src/models_params.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False

def _resolve_dataset_path(tsv_path: str) -> str:
  resolved = os.path.normpath(tsv_path)
  if not os.path.exists(resolved):
    raise FileNotFoundError(
      f"Training TSV not found: {resolved}. Expected the merged dataset at '{DEFAULT_TRAIN_TSV}'."
    )
  return resolved

# Focal loss for binary classification with logits
def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
  """
  Compute focal loss for binary classification with logits.
  Args:
    inputs: Raw model logits.
    targets: Binary targets (0 or 1), same shape as inputs.
    alpha: Weighting factor for positive examples.
    gamma: Focusing parameter to down-weight easy examples.
    reduction: 'mean', 'sum', or 'none'.
  """
  bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
  p_t = torch.exp(-bce)
  loss = alpha * (1 - p_t) ** gamma * bce
  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  else:
    return loss

def get_dataset_pretrain(tsv_path, quiet):
  """
  Load full graph for multi-relational pretraining.
  Returns same outputs as get_dataset, but using all relations as positives.
  """
  # Load the same edge_index and features
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet)
  ent2id, all_nodes_per_type = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id = rel2id_offset(edge_index)

  # Convert to indexed triplets
  indexed_edge_index = edge_ind_to_id(edge_index, ent2id, relation2id)

  # Use all triplets as positives
  all_triplets = indexed_edge_index.loc[:, ["head","interaction","tail"]].values

  # Split into train/val/test
  train_triplets, val_triplets, test_triplets = triple_sampling(
    all_triplets, 0.1, 0.2, quiet, BASE_SEED
  )

  # Build full message-passing graph from all edges (undirected + self-loops)
  num_nodes = sum(len(v) for v in all_nodes_per_type.values())
  full_index = graph_to_undirect(all_triplets, len(relation2id))
  full_index = add_self_loops(full_index, num_nodes, len(relation2id))

  # Flatten features, move to tensors/DEV
  train_triplets = torch.tensor(train_triplets).to(device)
  val_triplets   = torch.tensor(val_triplets).to(device)
  test_triplets  = torch.tensor(test_triplets).to(device)

  train_val_triplets = torch.cat([train_triplets, val_triplets], 0).to(device)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], 0).to(device)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  in_channels_dict = {nt:(feat.shape[1] if feat is not None else None)
                      for nt, feat in flattened_features_per_type.items()}
  
  num_nodes_per_type = {node_type:len(nodes) for node_type,nodes in all_nodes_per_type.items()}

  return (in_channels_dict, num_nodes_per_type, num_nodes, len(relation2id),
          train_triplets, full_index, flattened_features_per_type,
          val_triplets, train_val_triplets, test_triplets,
          train_val_test_triplets, edge_index, ent2id, relation2id)

def get_dataset(tsv_path, task, validation_size, test_size, quiet, seed, oversample_rate, undersample_rate):
  edge_index, node_features_per_type = load_data(tsv_path, {}, quiet, 
                                                #  debug=DEBUG
                                                 )
  
  # Label target edges BEFORE converting to IDs (need string entity names to derive edge types)
  edge_index = set_target_label(edge_index, [ x for x in task.split(',')] )
  
  ent2id, all_nodes_per_type  = entities2id_offset(edge_index, node_features_per_type, quiet)
  relation2id                 = rel2id_offset(edge_index)
  indexed_edge_index          = edge_ind_to_id(edge_index, ent2id, relation2id)
  flattened_features_per_type = entities_features_flattening(node_features_per_type, all_nodes_per_type)
  
  # Labels are already in edge_index, copy to indexed version
  indexed_edge_index["label"] = edge_index["label"].values

  non_target_triplets, target_triplets        = select_target_triplets(indexed_edge_index)
  train_triplets, val_triplets, test_triplets = triple_sampling(
                                                                target_triplets.loc[:,["head","interaction","tail"]].values, 
                                                                validation_size, 
                                                                test_size, 
                                                                quiet,
                                                                seed
                                                              )
  
  # under-sampling non-target
  if undersample_rate < 1.0:
    rnd = np.random.RandomState(seed)
    keep = int(len(non_target_triplets) * undersample_rate)
    idx = rnd.choice(len(non_target_triplets), size=keep, replace=False)
    non_target_triplets = non_target_triplets.iloc[idx]

  # over-sampling training positive edges
  if oversample_rate > 1:
    train_triplets = np.repeat(train_triplets, oversample_rate, axis=0)

  non_target_triplets = non_target_triplets.loc[:,["head","interaction","tail"]].values

  # print(f" non target {non_target_triplets.shape}")
  train_index = np.concatenate([non_target_triplets,train_triplets], axis=0)
  
  # print(f" target triplets{target_triplets.shape}")
  num_nodes_per_type = {node_type:len(nodes) for node_type,nodes in all_nodes_per_type.items()}

  num_entities = 0
  for features in num_nodes_per_type.values():
      num_entities += features

  # print(f" non target+target {train_index.shape}")
  train_index = graph_to_undirect(train_index, len(relation2id))
  # print(f" undirected {train_index.shape}")
  train_index = add_self_loops(train_index, num_entities, len(relation2id))
  # print(f" loops {train_index.shape}")
  
  train_triplets, val_triplets, test_triplets = torch.tensor(train_triplets), torch.tensor(val_triplets), torch.tensor(test_triplets)
  
  train_val_triplets = torch.cat([train_triplets, val_triplets], axis=0)
  train_val_test_triplets = torch.cat([train_val_triplets, test_triplets], axis=0)

  train_val_triplets = train_val_triplets.to(device)
  train_val_test_triplets = train_val_test_triplets.to(device)
        
  in_channels_dict = {node_type:(features.shape[1] if features!= None else None) for node_type, features in flattened_features_per_type.items() }
  num_relations = len(relation2id)

  return in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
    train_triplets, train_index, flattened_features_per_type, val_triplets, \
    train_val_triplets, test_triplets, train_val_test_triplets, edge_index, \
    ent2id, relation2id

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



def train(model, optimizer, gradnorm, reg_param, x_dict, index , target_triplets, target_labels, alpha, gamma, alpha_adv, change_points=None):
  metrics = {"Auroc":{},"Auprc":{},"Loss":{}}
  model.train()
  optimizer.zero_grad()
  if change_points != None:
    out = model(x_dict, index, change_points)
  else:
    out = model(x_dict, index)
  
  scores  = model.distmult(out, target_triplets)

  # loss    = model.score_loss(scores, target_labels)
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

  metrics["Loss"]   = loss.item()
  metrics["Auroc"]  = binary_auroc(scores, target_labels).item()
  metrics["Auprc"]  = binary_auprc(scores, target_labels).item()

  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), gradnorm)
  optimizer.step()

  return metrics

def test(model, reg_param, x_dict, index , target_triplets, target_labels, train_val_triplets, alpha, gamma, alpha_adv, change_points=None):    
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
from src.utils import get_edge_type

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

def main(model_name, dataset_tsv, task, runs, epochs, patience, validation_size, test_size, quiet, \
  evaluate_every, negative_rate, model_save_path, oversample_rate, undersample_rate, \
  pretrain_epochs, freeze_base, alpha, gamma, alpha_adv):
  all_run_metrics = []

  # select negative sampling function based on --alone flag
  if args.negative_sampling == 'filtered':
    print("[i] Using filtered negative sampling (only non-target edges) for training.")
    negative_sampling = negative_sampling_filtered
  else:
    print("[i] Using standard negative sampling (all edges) for training.")
    negative_sampling = negative_sampling

  # Multi-relational pretraining phase
  model_state = None
  if pretrain_epochs > 0:
      print(f'[i] Pretraining for {pretrain_epochs} epochs on all relations...')

      # Prepare pretrain dataset
      (pt_in_channels, pt_num_nodes_per_type, pt_num_entities, pt_num_relations,
       pt_train_triplets, pt_full_index, pt_flattened_feats,
        _, _, _, _, _, _, _) = get_dataset_pretrain(dataset_tsv, quiet)
      
      # Initialize model
      pre_model, lr, regularization, grad_norm = get_model(model_name, task, pt_in_channels, pt_num_nodes_per_type, pt_num_entities, pt_num_relations)
      pre_model = pre_model.to(device)

      # Pretraining loop
      optimizer_pre = torch.optim.Adam(pre_model.parameters(), lr=lr)

      change_points_pre = None
      if model_name == 'rgat':
        # pt_full_index è un numpy array [src, rel, dst]
        rel_ids = torch.from_numpy(pt_full_index[:,1]).to(device)
        change_points_pre = torch.cat([
            torch.tensor([0], device=device),
            (rel_ids[1:] != rel_ids[:-1]).nonzero(as_tuple=False).view(-1) + 1,
            torch.tensor([rel_ids.size(0)], device=device)
        ])

      pt_full_index = torch.tensor(pt_full_index).to(device)
      pt_flattened_feats = {node_type:(features.to(device) if features is not None else None) 
                      for node_type, features in pt_flattened_feats.items()}

      for epoch in tqdm(range(1, pretrain_epochs+1)):
          # negative sampling on all relations
          neg_triplets, neg_labels = negative_sampling(pt_train_triplets.cpu(), negative_rate)
          neg_triplets, neg_labels = neg_triplets.to(device), neg_labels.to(device)

          train(pre_model, optimizer_pre, grad_norm, \
                regularization, pt_flattened_feats, \
                pt_full_index, neg_triplets, neg_labels, \
                alpha, gamma, alpha_adv, change_points_pre)

      # Save pretrained weights
      pretrain_path = model_save_path.replace('.pt', '_pretrained.pt')
      torch.save(pre_model.state_dict(), pretrain_path)
      print(f'[i] Pretrained model saved to {pretrain_path}')
      # For fine-tuning: load weights
      model_state = torch.load(pretrain_path)

  for i in range(runs):
    run_model_save_path = model_save_path.replace(".pt", f"_run{i}.pt")
    random_seed = BASE_SEED + i
    # set_seed(random_seed)
    print(f'[i] Using random seed {random_seed}')
    
    # Get dataset
    print('[i] Getting dataset...', end='', flush=True)
    start_time_dataset = time.time()

    in_channels_dict, num_nodes_per_type, num_entities, num_relations, \
    train_triplets, train_index, flattened_features_per_type, val_triplets, \
    train_val_triplets, test_triplets, train_val_test_triplets, \
    edge_index, ent2id, relation2id = get_dataset(dataset_tsv, task, validation_size, test_size, quiet, \
                            random_seed, oversample_rate, undersample_rate)
    
    end_time_dataset = round(time.time() - start_time_dataset, 2)
    # print(f'ok ({end_time_dataset} seconds)')
    
    # Model definition
    model, lr, regularization, grad_norm = get_model(model_name, task, in_channels_dict, num_nodes_per_type, num_entities, num_relations)

    # Move to device
    # model                       = model.to(device)
    # train_index                 = train_index.to(device)
    if model_state is not None:
      model.load_state_dict(model_state, strict=False)
      if freeze_base:
          for name, param in model.named_parameters():
              if 'conv_layers' in name or 'relation_embeddings_per_layer' in name:
                  param.requires_grad = False
    model = model.to(device)
    train_index = train_index.to(device)
    flattened_features_per_type = {node_type:(features.to(device) if features!= None else None) for node_type,features in flattened_features_per_type.items()}
    
    change_points = None
    if model_name == 'rgat':
      rel_ids_sorted = train_index[:, 1]
      change_points = torch.cat([
          torch.tensor([0], device=device),
          (rel_ids_sorted[1:] != rel_ids_sorted[:-1]).nonzero(as_tuple=False).view(-1) + 1,
          torch.tensor([rel_ids_sorted.size(0)], device=device) 
      ])
      
    # Optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training runs
    torch.autograd.set_detect_anomaly(True)
    val_metrics       = {"Auroc":0, "Auprc":0, "Loss":0, "MRR":0, "Hits@":0}
    patience_trigger  = 0
    best_mixed_metric = 0
    best_model_found  = False
    with trange(1, (epochs + 1), desc=f'Run {i} | Epochs', position=0) as epochs_tqdm:
      for epoch in epochs_tqdm:
        training_triplets, train_labels = negative_sampling(train_triplets, negative_rate)
        training_triplets, train_labels = training_triplets.to(device), train_labels.to(device)
        # Train
        train_metrics = train(
            model,
            optimizer, 
            grad_norm, 
            regularization,
            flattened_features_per_type,
            train_index,
            training_triplets,
            train_labels,
            alpha,
            gamma,
            alpha_adv,
            change_points
          )
        # Validate
        if epoch%evaluate_every==0:
          validation_triplets, val_labels = negative_sampling(val_triplets, negative_rate)
          validation_triplets, val_labels = validation_triplets.to(device), val_labels.to(device)
          val_metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            validation_triplets,
            val_labels,
            train_val_triplets,
            alpha,
            gamma,
            alpha_adv,
            change_points
          )
        
        mixed_metric = 0.2*val_metrics["Auroc"] + 0.4*val_metrics["Auprc"] + 0.4*val_metrics["MRR"]
        if mixed_metric > best_mixed_metric:
          best_mixed_metric = mixed_metric
          patience_trigger = 0
          # model is saved only if validation improves
          torch.save(model.state_dict(), run_model_save_path)
          print("[i] Best model updated.")
          best_model_found = True
        else:
          patience_trigger += 1
        if patience_trigger > patience:
          break
        
        epochs_tqdm.set_postfix(loss=train_metrics["Loss"], Tr_Auroc=train_metrics["Auroc"], Tr_Auprc=train_metrics["Auprc"],
                                Val_Auroc=val_metrics["Auroc"], Val_Auprc=val_metrics["Auprc"], Val_mrr=val_metrics["MRR"], Val_hits=val_metrics["Hits@"],
                                metric=mixed_metric, best_metric=best_mixed_metric)
      
      # Load best model
      if best_model_found:
        model.load_state_dict(torch.load(run_model_save_path))
      # Test best model
      model.eval()
      with torch.no_grad():    
          testing_triplets, test_labels = negative_sampling(test_triplets, negative_rate)
          testing_triplets, test_labels = testing_triplets.to(device), test_labels.to(device)
          metrics = test(
            model,
            regularization,
            flattened_features_per_type,
            train_index,
            testing_triplets,
            test_labels,
            train_val_test_triplets,
            alpha,
            gamma,
            alpha_adv,
            change_points
          )  
      print(f"Run {i} | Test Auroc: {metrics['Auroc']:.3f}, Test Auprc: {metrics['Auprc']:.3f}, Test MRR: {metrics['MRR']:.3f}, TEST HITS: {metrics['Hits@']}")
      all_run_metrics.append({
          "Auroc": metrics["Auroc"],
          "Auprc": metrics["Auprc"],
          "MRR": metrics["MRR"],
          "Hits@1": metrics["Hits@"][1],
          "Hits@3": metrics["Hits@"][3],
          "Hits@10": metrics["Hits@"][10]
      })
      print(f"[i] Completed run {i}/{args.runs}")

      # print(f"[i] Saving best model of run {i} to {run_model_save_path}")

      print(f"[i] Evaluating model on entire dataset for ranking...")
      # Eval model on entire dataset and get ranking
      rank = eval(model, flattened_features_per_type, train_index, edge_index, ent2id, relation2id, change_points, task=task)
      # Show better print predictions
      top_p_num = min(10, len(rank))
      print(f"\n[i] Top {top_p_num} predictions:")
      for idx, pred in enumerate(rank[:top_p_num], 1):
          print(f"  {idx}. Head: {pred['head']:30s} -> Tail: {pred['tail']:20s} | Confidence: {pred['confidence']:.4f}")
      # Save ranking to JSON
      ranking_save_path = run_model_save_path.replace(".pt", "_ranking.json")
      with open(ranking_save_path, "w") as f:
          json.dump(rank, f, indent=4)
      print(f"[i] Saved ranking to {ranking_save_path}")


  metric_keys = ["Auroc", "Auprc", "MRR", "Hits@1", "Hits@3", "Hits@10"]
  avg_metrics = {key: float(np.mean([m[key] for m in all_run_metrics])) for key in metric_keys}
  std_metrics = {key: float(np.std([m[key] for m in all_run_metrics])) for key in metric_keys}

  # Salvataggio su JSON
  result_to_save = {
      "individual_runs": all_run_metrics,
      "average_metrics": avg_metrics,
      "std_metrics": std_metrics
  }
  json_save_path = model_save_path.replace(".pt", "_metrics.json")
  with open(json_save_path, "w") as f:
      json.dump(result_to_save, f, indent=4)
  # print(f"Saved metrics to {json_save_path}")

  return None

if __name__ == '__main__':
  print(f'[i] Running on {device}')

  parser = argparse.ArgumentParser(description='Ablation study on Vitagraph generation process.')
  # add 

  parser.add_argument('-t', '--target', type=str, default=None,
                      help='[DEPRECATED] Ignored. Training always uses the merged TSV (see --tsv).')
  parser.add_argument('--tsv', type=str, default=DEFAULT_TRAIN_TSV,
                      help=f"Path to the training TSV (default: {DEFAULT_TRAIN_TSV})")
  parser.add_argument('-m', '--model', type=str, default='compgcn', choices=['rgcn', 'rgat', 'compgcn'], help='Model to use for the ablation study.')
  parser.add_argument('-r', '--runs', type=int, default=1, help='Number of runs for the ablation study.')
  parser.add_argument('-e', '--epochs', type=int, default=400, help='Number of epochs for the ablation study.')
  parser.add_argument('-p', '--patience', type=int, default=200, help='Patience trigger.')
  parser.add_argument('--validation_size', type=float, default=0.1, help='Validation size for the ablation study.')
  parser.add_argument('--test_size', type=float, default=0.2, help='Test size for the ablation study.')
  parser.add_argument('--quiet', action='store_true', help='If set, the ablation study will print debug output.')
  parser.add_argument('--evaluate_every', type=int, default=5, help='Evaluate every n epochs.')
  parser.add_argument('--negative_rate', type=float, default=1, help='Negative sampling rate for the ablation study.')
  parser.add_argument('-a', '--alone', action='store_true',
                      help='[DEPRECATED] Ignored. Dataset selection is controlled by --tsv.')
  parser.add_argument('--oversample_rate', type=int, default=5, help='how many times to repeat the positive training triplets')
  parser.add_argument('--undersample_rate', type=float, default=0.5, help='fraction [0,1] of non-target triplets to keep in the training graph')
  parser.add_argument('--pretrain_epochs', type=int, default=0, help='Number of epochs for multi-relational pretraining')
  parser.add_argument('--freeze_base', action='store_true', help='Freeze pre-trained conv layers during fine-tuning')
  parser.add_argument('--alpha', type=float, default=0.25, help='Alpha value of the focal loss')
  parser.add_argument('--gamma', type=float, default=3.0, help='Gamma value of the focal loss')
  parser.add_argument('--alpha_adv', type=float, default=2.0, help='Alpha value for the hard-negative mining loss'), 
  parser.add_argument('--negative_sampling', type=str, default='base', help='')  

  # add task as argument
  parser.add_argument('--task', type=str, 
                      #default='Compound-ExtGene', 
                      default='TARGET', 
                      help='Task to perform. Could be a comma-separated list of edge types or interaction names (e.g., "CMP_BIND,ENZYME"). If not specified, defaults to "TARGET".')


  args            = parser.parse_args()
  model           = args.model
  runs            = args.runs
  epochs          = args.epochs
  patience        = args.patience
  validation_size = args.validation_size
  test_size       = args.test_size
  quiet           = args.quiet
  evaluate_every  = args.evaluate_every
  negative_rate   = args.negative_rate
  dataset_tsv     = _resolve_dataset_path(args.tsv)
  oversample_rate = args.oversample_rate
  undersample_rate = args.undersample_rate
  pretrain_epochs = args.pretrain_epochs
  freeze_base     = args.freeze_base
  alpha           = args.alpha
  gamma           = args.gamma
  alpha_adv       = args.alpha_adv

  # get dataset_name (use os.path for cross-platform compatibility)
  dataset_basename = os.path.basename(args.tsv)
  dataset_name = os.path.splitext(dataset_basename)[0]
  print(f'[i] Using dataset: {dataset_name}')


  time_stamp      = time.strftime('%Y%m%d_%H%M%S')
  task            = args.task
  task_clean      = task.lower().replace('-', '_').replace(',', '_')
  folder_name = task_clean + '_' + dataset_name + '_' + time_stamp
  model_save_dir = os.path.join('models', folder_name)
  os.makedirs(model_save_dir, exist_ok=True)
  model_save_path = os.path.join(model_save_dir, f'{model.lower()}.pt')

  # save parameters used for the run
  params_save_path = model_save_path.replace('.pt', '_params.json')
  with open(params_save_path, 'w') as f:
      json.dump(vars(args), f, indent=4)


  if not quiet: print(f'Running training of model: {model}, runs: {runs} | device: {device}')
  if (args.target is not None) and (not quiet):
    print('[i] Note: --target is deprecated and ignored (using merged TSV).')
  if args.alone and (not quiet):
    print('[i] Note: --alone is deprecated and ignored (using merged TSV).')


  """
  Example commands:
  python train_and_eval.py --model rgcn --epochs 1
  python train_and_eval.py --model rgcn --runs 3 --epochs 100
  python train_and_eval.py --model rgcn --runs 3 --epochs 100
  python train_and_eval.py --model compgcn --epochs 100 --negative_sampling filtered

  python train_and_eval.py --model compgcn --pretrain_epochs 100 --freeze_base --epochs 200

  -- on DRKG (simple triplets):
  python train_and_eval.py --model compgcn --epochs 100 --task CMP_BIND --tsv dataset/drkg/drkg_reduced.zip 
  
  -- on DRKG dataset:
  python train_and_eval.py --model compgcn --epochs 300 --task Compound-Gene --tsv dataset/drkg/drkg.tsv
  
  -- on DRKG_reduced dataset:
  python train_and_eval.py --model compgcn --epochs 300 --task Compound-Gene --tsv dataset/drkg/drkg_reduced.tsv
  python train_and_eval.py --model compgcn --epochs 3 --task Compound-Gene --tsv dataset/drkg/drkg_reduced.tsv

  """

  main(model, dataset_tsv, task, runs, epochs, patience, validation_size, test_size, quiet, \
      evaluate_every, negative_rate, model_save_path, oversample_rate, undersample_rate, \
      pretrain_epochs, freeze_base, alpha, gamma, alpha_adv)