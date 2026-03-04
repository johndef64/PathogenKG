"""
Evaluation metrics per Knowledge Graph Link Prediction — Filtered Setting.

Queste metriche sono standard nella letteratura KGE e forniscono una valutazione più realistica

Queste metriche sono full graph, non campionate, e possono essere computazionalmente intensive su grafi molto grandi. Tuttavia, sono più affidabili rispetto a metriche campionate o non filtrate, specialmente in contesti con molti positivi noti (come i KGE biomedici).

Versione corretta e migliorata rispetto alla proposta originale.
Cambiamenti rispetto alla versione proposta:
  1. Aggiunto check di sicurezza per evitare rank=0 e divisione per zero nel MRR
  2. Aggiunto clamp del rank minimo a 1 (difesa contro edge case)
  3. Separazione metriche tail/head per diagnostica
  4. Logging opzionale per debug
  5. Supporto per device mismatch
  6. Gestione edge case: nodo positivo assente da unique_nodes


"""

import torch
from collections import defaultdict


def build_positive_maps(all_target_triplets):
    """
    Costruisce le mappe dei positivi noti per il filtered setting.
    
    Args:
        all_target_triplets: tensor (N, 3) con TUTTE le triple positive 
                             (train + val + test) della relazione target.
    
    Returns:
        all_positives_tail: dict (h, r) -> set of t
        all_positives_head: dict (r, t) -> set of h
    """
    all_positives_tail = defaultdict(set)
    all_positives_head = defaultdict(set)
    
    for i in range(all_target_triplets.size(0)):
        h = all_target_triplets[i, 0].item()
        r = all_target_triplets[i, 1].item()
        t = all_target_triplets[i, 2].item()
        all_positives_tail[(h, r)].add(t)
        all_positives_head[(r, t)].add(h)
    
    return all_positives_tail, all_positives_head


def evaluation_metrics_filtered(
    model, 
    embeddings, 
    all_target_triplets,  # train + val + test della relazione target
    test_triplets,        # solo le triple di test
    all_graph_nodes,      # tutti i nodi unici del grafo
    device, 
    hits_k=[1, 3, 10],
    verbose=False
):
    """
    Calcola MRR e Hits@K con filtered setting (standard nella letteratura KGE).
    
    Il filtered setting rimuove dal ranking tutti i veri positivi noti
    (tranne quello che si sta valutando), evitando di penalizzare il modello
    per aver assegnato score alti ad altre risposte corrette.
    
    Args:
        model: modello con metodo .distmult(embeddings, triplets) -> scores
        embeddings: embedding dei nodi (output del GNN encoder)
        all_target_triplets: tensor (N, 3) — TUTTE le triple positive (train+val+test)
        test_triplets: tensor (M, 3) — solo le triple di test da valutare
        all_graph_nodes: tensor (E,) — tutti i nodi unici del grafo
        device: torch device
        hits_k: lista di K per Hits@K (default [1, 3, 10])
        verbose: se True, stampa info di debug ogni 100 triple
    
    Returns:
        dict con chiavi: 'mrr', 'mrr_tail', 'mrr_head', 
                         'hits@K' per ogni K, 'hits@K_tail', 'hits@K_head'
    """
    model.eval()
    unique_nodes = all_graph_nodes.to(device)
    num_entities = unique_nodes.size(0)
    
    # Mappa nodo -> indice nel vettore unique_nodes
    node_to_idx = {}
    for i in range(num_entities):
        node_to_idx[unique_nodes[i].item()] = i
    
    # Costruisci mappe dei positivi per il filtering
    all_positives_tail, all_positives_head = build_positive_maps(all_target_triplets)
    
    tail_ranks = []
    head_ranks = []
    skipped = 0
    
    with torch.no_grad():
        for i in range(test_triplets.size(0)):
            h, r, t = test_triplets[i]
            h_i, r_i, t_i = h.item(), r.item(), t.item()
            
            # Verifica che entrambi i nodi siano nel grafo
            if h_i not in node_to_idx or t_i not in node_to_idx:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] Tripla {i}: nodo mancante da unique_nodes")
                continue
            
            # ============================================
            # TAIL PREDICTION: dato (h, r, ?), ranking di t
            # ============================================
            tail_candidates = torch.stack([
                h.expand(num_entities).to(device),
                r.expand(num_entities).to(device),
                unique_nodes
            ], dim=1)
            tail_scores = model.distmult(embeddings, tail_candidates)
            
            # Score del vero positivo
            true_tail_score = tail_scores[node_to_idx[t_i]]
            
            # Filtered: maschera i positivi noti tranne t_i
            filter_mask = torch.ones(num_entities, dtype=torch.bool, device=device)
            for known_t in all_positives_tail.get((h_i, r_i), set()):
                if known_t != t_i and known_t in node_to_idx:
                    filter_mask[node_to_idx[known_t]] = False
            
            # Il positivo t_i NON viene mascherato (la condizione known_t != t_i lo protegge)
            # Quindi true_tail_score è incluso in filtered_scores
            filtered_scores = tail_scores[filter_mask]
            
            # Rank = quanti score sono >= al positivo (incluso se stesso, quindi rank minimo = 1)
            tail_rank = (filtered_scores >= true_tail_score).sum().item()
            
            # Safety: rank deve essere almeno 1 (difesa contro edge case numerici)
            tail_rank = max(tail_rank, 1)
            tail_ranks.append(tail_rank)
            
            # ============================================
            # HEAD PREDICTION: dato (?, r, t), ranking di h
            # ============================================
            head_candidates = torch.stack([
                unique_nodes,
                r.expand(num_entities).to(device),
                t.expand(num_entities).to(device)
            ], dim=1)
            head_scores = model.distmult(embeddings, head_candidates)
            
            true_head_score = head_scores[node_to_idx[h_i]]
            
            filter_mask = torch.ones(num_entities, dtype=torch.bool, device=device)
            for known_h in all_positives_head.get((r_i, t_i), set()):
                if known_h != h_i and known_h in node_to_idx:
                    filter_mask[node_to_idx[known_h]] = False
            
            filtered_scores = head_scores[filter_mask]
            head_rank = (filtered_scores >= true_head_score).sum().item()
            head_rank = max(head_rank, 1)
            head_ranks.append(head_rank)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Valutate {i+1}/{test_triplets.size(0)} triple "
                      f"(tail_rank={tail_rank}, head_rank={head_rank})")
    
    if skipped > 0:
        print(f"  [WARN] {skipped} triple saltate per nodi mancanti")
    
    if len(tail_ranks) == 0:
        print("  [ERROR] Nessuna tripla valutata!")
        return {
            'mrr': 0.0, 'mrr_tail': 0.0, 'mrr_head': 0.0,
            **{f'hits@{k}': 0.0 for k in hits_k},
            **{f'hits@{k}_tail': 0.0 for k in hits_k},
            **{f'hits@{k}_head': 0.0 for k in hits_k},
            'num_evaluated': 0, 'num_skipped': skipped
        }
    
    # Converti in tensori
    tail_ranks_t = torch.tensor(tail_ranks, dtype=torch.float, device=device)
    head_ranks_t = torch.tensor(head_ranks, dtype=torch.float, device=device)
    all_ranks_t = torch.cat([tail_ranks_t, head_ranks_t])
    
    # Calcola metriche
    results = {
        'mrr': (1.0 / all_ranks_t).mean().item(),
        'mrr_tail': (1.0 / tail_ranks_t).mean().item(),
        'mrr_head': (1.0 / head_ranks_t).mean().item(),
        'num_evaluated': len(tail_ranks),
        'num_skipped': skipped,
    }
    
    for k in hits_k:
        results[f'hits@{k}'] = (all_ranks_t <= k).float().mean().item()
        results[f'hits@{k}_tail'] = (tail_ranks_t <= k).float().mean().item()
        results[f'hits@{k}_head'] = (head_ranks_t <= k).float().mean().item()
    
    return results


# ============================================
# Funzione helper per stampare i risultati
# ============================================
def print_metrics(results, title="Evaluation Results"):
    """Stampa le metriche in formato leggibile."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Triple valutate: {results['num_evaluated']}"
          f" (saltate: {results['num_skipped']})")
    print(f"  MRR (overall):   {results['mrr']:.4f}")
    print(f"  MRR (tail):      {results['mrr_tail']:.4f}")
    print(f"  MRR (head):      {results['mrr_head']:.4f}")
    for key in sorted(results.keys()):
        if key.startswith('hits@') and '_' not in key:
            k = key.replace('hits@', '')
            print(f"  Hits@{k} (overall): {results[key]:.4f}")
            print(f"  Hits@{k} (tail):    {results.get(f'hits@{k}_tail', 0):.4f}")
            print(f"  Hits@{k} (head):    {results.get(f'hits@{k}_head', 0):.4f}")
    print(f"{'='*50}\n")


# ============================================
# Esempio di utilizzo
# ============================================
"""
# 1. Raccogli TUTTE le triple della relazione target (train + val + test)
all_target_triplets = torch.cat([
    train_target_triplets,
    val_target_triplets,
    test_target_triplets
], dim=0)

# 2. Raccogli tutti i nodi unici del grafo
all_graph_nodes = torch.unique(torch.cat([
    graph.edge_index[0], 
    graph.edge_index[1]
]))

# 3. Valuta
results = evaluation_metrics_filtered(
    model=model,
    embeddings=embeddings,
    all_target_triplets=all_target_triplets,
    test_triplets=test_target_triplets,
    all_graph_nodes=all_graph_nodes,
    device=device,
    hits_k=[1, 3, 10],
    verbose=True
)

print_metrics(results, title="PathogenKG — Filtered Setting")
"""
