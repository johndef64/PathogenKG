## Heterogeneous Graph Convolutional Networks (HGCNs)

Heterogeneous Graph Convolutional Networks (HGCNs) extend traditional GCNs to handle heterogeneous graphs, which contain multiple types of nodes and edges. This is particularly useful in knowledge graphs, where entities and relationships can vary significantly.

### Key Concepts

1. **Heterogeneous Graphs**: Graphs with different types of nodes and edges. For example, in a knowledge graph, you might have "Person", "Organization", and "Location" as node types, and "works_at", "located_in" as edge types.

2. **Node and Edge Types**: In HGCNs, the model must account for the different types of nodes and edges. This is often done by using type-specific embeddings and aggregation functions.

3. **Message Passing**: Similar to traditional GCNs, HGCNs use a message passing framework, but with modifications to handle heterogeneity. Messages are aggregated based on the types of neighboring nodes and edges.

### Applications

HGCNs are particularly useful in scenarios where the graph structure is complex and involves multiple entity types. Some applications include:

- **Knowledge Graph Completion**: Predicting missing links in a knowledge graph by leveraging the relationships between different entity types.
- **Recommendation Systems**: Using heterogeneous graphs to model user-item interactions, where users and items can belong to different categories.
- **Social Network Analysis**: Analyzing relationships in social networks where users, posts, and comments are different types of nodes.

### Challenges

1. **Scalability**: HGCNs can be computationally intensive, especially for large graphs with many node and edge types.

2. **Interpretability**: Understanding the model's decisions can be more challenging in heterogeneous settings due to the complexity of the graph structure.

3. **Data Sparsity**: Heterogeneous graphs may suffer from sparsity issues, particularly for less common node and edge types.

### Conclusion

Heterogeneous Graph Convolutional Networks represent a powerful extension of traditional GCNs, enabling the modeling of complex relationships in knowledge graphs and other heterogeneous data structures. As research in this area continues to evolve, we can expect to see more sophisticated models and applications emerge.

## Architettura HeterogeneousCompGCN: Spiegazione Dettagliata

L'architettura **HeterogeneousCompGCN** è una rete neurale specializzata per elaborare grafi di conoscenza eterogenei. Ecco una spiegazione passo-passo di come funziona:

## Diagramma dell'Architettura

```
                    HETEROGENEOUS COMPGCN ARCHITECTURE
                    ===================================

INPUT LAYER:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Persone       │  │     Città       │  │    Aziende      │
│ [età, altezza]  │  │  [solo nome]    │  │[dip., fatturato]│
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   MLP (3→128)   │  │Learned Embedding│  │   MLP (2→128)   │
│   + Dropout     │  │   (ID→128)      │  │   + Dropout     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ UNIFORM EMBEDDINGS  │
                    │   [128 dimensioni]  │
                    └─────────────────────┘

COMPGCN LAYERS (ripetuto per N layers):
┌─────────────────────────────────────────────────────────────────┐
│                        COMPGCN LAYER                           │
│                                                                 │
│  Node Embeddings     Relation Embeddings                       │
│  ┌─────────────┐     ┌─────────────────┐                       │
│  │    Mario    │ ◄─► │   lavora_a      │                       │
│  │   Milano    │     │ si_trova_in     │                       │
│  │   Bocconi   │     │ha_dipendente    │                       │
│  └─────────────┘     └─────────────────┘                       │
│          │                    │                                │
│          ▼                    ▼                                │
│  ┌─────────────────────────────────┐                           │
│  │   COMPOSITIONAL OPERATION      │                           │
│  │   • mult: ent ⊗ rel            │                           │
│  │   • sub:  ent - rel            │                           │
│  │   • corr: correlation(ent,rel) │                           │
│  └─────────────────────────────────┘                           │
│          │                                                     │
│          ▼                                                     │
│  ┌─────────────────────────────────┐                           │
│  │      MESSAGE PASSING           │                           │
│  │                                 │                           │
│  │  Forward:  src ─rel→ dst        │                           │
│  │  Backward: dst ─rel⁻¹→ src      │                           │
│  │  Self-loop: node ─loop→ node    │                           │
│  └─────────────────────────────────┘                           │
│          │                                                     │
│          ▼                                                     │
│  ┌─────────────────────────────────┐                           │
│  │       AGGREGATION              │                           │
│  │ (Forward + Backward + Loop)/3   │                           │
│  │    + Edge Normalization         │                           │
│  └─────────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACTIVATION & NORMALIZATION                  │
│              ReLU/GELU + LayerNorm (opzionale)                 │
└─────────────────────────────────────────────────────────────────┘

OUTPUT LAYER:
┌─────────────────────────────────────────────────────────────────┐
│                     SCORING FUNCTIONS                          │
│                                                                 │
│  DistMult:                    ComplEx:                         │
│  ┌─────────────────────┐     ┌─────────────────────┐           │
│  │ score = Σ(s⊗r⊗o)   │     │ score = Re(s⊗r⊗ō)  │           │
│  │                     │     │ (numeri complessi)  │           │
│  └─────────────────────┘     └─────────────────────┘           │
│          │                            │                        │
│          └────────────┬───────────────┘                        │
│                       ▼                                        │
│              ┌─────────────────┐                               │
│              │ PROBABILITÀ     │                               │
│              │ (0 ≤ p ≤ 1)     │                               │
│              └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘

LEARNING:
┌─────────────────────────────────────────────────────────────────┐
│                      LOSS FUNCTIONS                            │
│                                                                 │
│  Score Loss:                  Regularization Loss:             │
│  BCE(predictions, truth)      L2(embeddings)                   │
│           │                            │                       │
│           └────────────┬───────────────┘                       │
│                        ▼                                       │
│               ┌─────────────────┐                              │
│               │ BACKPROPAGATION │                              │
│               │ (aggiorna pesi) │                              │
│               └─────────────────┘                              │
│                        │                                       │
│                        ▼                                       │
│               ┌─────────────────┐                              │
│               │ PARAMETER UPDATE│                              │
│               │ ∇w → w_new      │                              │
│               └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                         │
                         │ (aggiorna tutti i pesi del network)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ◄── FEEDBACK LOOP ──►                       │
│                                                                 │
│  I gradienti aggiornano:                                        │
│  • Pesi delle MLP di input                                     │
│  • Embeddings delle relazioni                                  │
│  • Pesi delle trasformazioni CompGCN (w_in, w_out, w_loop)     │
│  • Parametri delle funzioni di scoring                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │     CICLO ITERATIVO DI APPRENDIMENTO                       │ │
│  │                                                             │ │
│  │  Epoca 1: Predizioni → Loss → Backprop → Update Pesi       │ │
│  │  Epoca 2: Predizioni → Loss → Backprop → Update Pesi       │ │
│  │  ...                                                        │ │
│  │  Epoca N: Convergenza                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

FLUSSO DATI: 
Forward:  Input → Uniformazione → CompGCN Layers → Scoring → Loss
Backward: Loss → Gradients → Parameter Updates → CompGCN Layers (ciclo)
```

### 1. **Preparazione dei Dati di Input**

**Cosa succede**: Il modello riceve due tipi di informazioni principali:
- **Nodi**: Entità del grafo (es. "Mario Rossi", "Milano", "Università Bocconi")
- **Archi**: Relazioni tra le entità (es. "lavora_a", "si_trova_in")

**Problema da risolvere**: I nodi possono essere di tipi diversi (persone, luoghi, organizzazioni) e avere caratteristiche diverse. Alcuni potrebbero avere features numeriche (età, coordinate GPS), altri solo un nome.

**Soluzione**: Il modello usa un **dizionario di MLPs** (Multi-Layer Perceptrons) per uniformare tutti i nodi:

#### Come funziona l'uniformazione?

**Scenario iniziale**: Abbiamo diversi tipi di nodi con informazioni diverse:
```
Persona "Mario":     [età: 35, altezza: 1.75, stipendio: 50000]  → 3 dimensioni
Città "Milano":      solo il nome                                 → 0 dimensioni numeriche  
Azienda "Bocconi":   [dipendenti: 2000, fatturato: 100M]         → 2 dimensioni
```

**Processo di uniformazione**:

1. **Per nodi CON features numeriche** (Persona, Azienda):
   ```
   Input: [35, 1.75, 50000]  (Mario)
   ↓ 
   MLP (rete neurale): Linear(3 → 128) + Dropout
   ↓
   Output: [0.23, -0.15, 0.67, ..., 0.89]  (128 numeri)
   ```

2. **Per nodi SENZA features** (Città):
   ```
   Input: ID numerico di Milano (es. 1542)
   ↓
   Embedding Table: tabella_città[1542] 
   ↓
   Output: [0.41, 0.88, -0.32, ..., 0.12]  (128 numeri learnable)
   ```

**Risultato finale**: Tutti i nodi, indipendentemente dal tipo, diventano vettori di 128 numeri:
```
Mario:   [0.23, -0.15, 0.67, ..., 0.89]  ← da features numeriche
Milano:  [0.41, 0.88, -0.32, ..., 0.12]  ← da embedding learnable  
Bocconi: [0.77, 0.34, -0.21, ..., 0.45]  ← da features numeriche
```

**Perché è importante?**: 
- **Compatibilità**: Tutti i nodi possono ora "parlare la stessa lingua" matematica
- **Operazioni uniformi**: Le operazioni successive (moltiplicazioni, somme) funzionano su tutti i tipi
- **Apprendimento condiviso**: Il modello può imparare pattern comuni tra diversi tipi di entità

#### Come si chiama questo approccio?

Questo processo di uniformazione ha diversi nomi tecnici:

1. **Heterogeneous Node Embedding**: Embedding per nodi eterogenei
2. **Multi-modal Feature Projection**: Proiezione di features multi-modali
3. **Type-specific Feature Transformation**: Trasformazione di features specifica per tipo

**Tecniche specifiche utilizzate**:

- **Learned Embeddings** (Embedding Appresi): 
  - Per nodi senza features (come le città)
  - Il modello impara automaticamente rappresentazioni ottimali
  - Tecnicamente: `nn.Embedding(num_entities, embedding_dim)`

- **Feature Projection MLPs** (MLP di Proiezione Features):
  - Per nodi con features numeriche (persone, aziende)
  - Trasforma features di dimensioni diverse in uno spazio comune
  - Tecnicamente: `nn.Linear(input_features, embedding_dim) + nn.Dropout()`

- **Type-aware Embedding**: 
  - Il modello "sa" che tipo di nodo sta processando
  - Usa trasformazioni diverse per tipi diversi
  - Mantiene un dizionario di trasformazioni: `nn.ModuleDict()`

**Analogia**: È come avere un traduttore universale che:
- Per chi parla inglese → usa un traduttore inglese-esperanto
- Per chi parla cinese → usa un traduttore cinese-esperanto  
- Per chi non parla nessuna lingua → gli insegna direttamente l'esperanto
- Risultato: tutti comunicano in "esperanto matematico" (embedding space)

### 2. **Gestione delle Relazioni**

**Cosa succede**: Per ogni livello della rete, il modello mantiene una tabella di embeddings per le relazioni.

**Perché è importante**: Le relazioni non sono solo etichette, ma hanno un "significato" che il modello deve imparare. Ad esempio:
- "lavora_a" ha un significato diverso da "è_nato_a"
- Il modello deve capire che se Mario "lavora_a" Milano e Milano "si_trova_in" Italia, allora c'è una connessione indiretta tra Mario e Italia

**Dettaglio tecnico**: Per ogni relazione, il modello crea anche la relazione inversa (es. se "Mario lavora_a Bocconi", allora "Bocconi ha_dipendente Mario").

### 3. **Compositional Operation (Operazione di Composizione)**

**Cosa succede**: Questa è l'operazione chiave che combina le informazioni di un nodo con quelle di una relazione.

**Analogia**: Immagina di avere una frase "Mario lavora_a". Il modello deve capire cosa significa questa combinazione. Può farlo in tre modi:
- **Moltiplicazione** (`mult`): Enfatizza gli aspetti comuni tra Mario e "lavora_a"
- **Sottrazione** (`sub`): Evidenzia le differenze o i cambiamenti
- **Correlazione** (`corr`): Trova pattern complessi di interazione

**Esempio pratico**:
```
Embedding di "Mario" = [0.5, 0.8, 0.2, ...]
Embedding di "lavora_a" = [0.3, 0.9, 0.1, ...]
Risultato (mult) = [0.15, 0.72, 0.02, ...] (moltiplicazione elemento per elemento)
```

### 4. **Message Passing (Passaggio di Messaggi)**

**Cosa succede**: Ogni nodo raccoglie informazioni dai suoi vicini attraverso tre canali:

#### a) **Forward Messages** (Messaggi in Avanti)
- Mario → lavora_a → Bocconi
- Il messaggio va da Mario verso Bocconi attraverso la relazione "lavora_a"

#### b) **Backward Messages** (Messaggi all'Indietro)  
- Bocconi → ha_dipendente → Mario
- Il messaggio va da Bocconi verso Mario attraverso la relazione inversa

#### c) **Self-Loop Messages** (Messaggi di Auto-connessione)
- Mario → se_stesso
- Permette al nodo di mantenere le sue informazioni originali

**Analogia**: È come se ogni persona in una rete sociale ricevesse informazioni da:
- Le persone a cui è connessa direttamente
- Le persone che sono connesse a lei
- I suoi pensieri/caratteristiche personali

### 5. **Aggregazione e Normalizzazione**

**Cosa succede**: I tre tipi di messaggi vengono combinati usando una media semplice:
```
Nuovo_embedding_Mario = (Messaggio_Forward + Messaggio_Backward + Messaggio_SelfLoop) / 3
```

**Normalizzazione per grado**: Se Mario è connesso a molte entità (alto grado), i suoi messaggi vengono "attenuati" per evitare che domini la rete.

**Analogia**: È come regolare il volume delle voci in una conversazione - chi parla troppo viene abbassato di volume.

### 6. **Layers Multipli**

**Cosa succede**: Il processo viene ripetuto per più livelli (layers), permettendo di catturare relazioni sempre più complesse.

**Esempio progressivo**:
- **Layer 1**: Mario sa di lavorare alla Bocconi
- **Layer 2**: Mario scopre che la Bocconi è a Milano (attraverso altri dipendenti)
- **Layer 3**: Mario scopre che Milano è in Italia (attraverso altre connessioni)

### 7. **Scoring e Predizione**

**Cosa succede**: Alla fine, il modello può predire nuove relazioni usando funzioni di scoring:

#### **DistMult**
```
Score(Mario, lavora_a, Google) = Somma(Embedding_Mario × Embedding_lavora_a × Embedding_Google)
```

#### **ComplEx**
Più complessa, usa numeri complessi per catturare pattern asimmetrici nelle relazioni.

**Risultato**: Un punteggio che indica quanto è probabile che "Mario lavora a Google" sia vero.

### 8. **Apprendimento**

**Cosa succede**: Il modello impara confrontando le sue predizioni con la realtà:
- **Loss di predizione**: Quanto sono sbagliate le predizioni
- **Loss di regolarizzazione**: Evita che il modello "memorizzi" invece di imparare pattern generali

### Riassunto del Flusso Completo

1. **Input** → Nodi eterogenei + Relazioni
2. **Uniformazione** → Tutti i nodi diventano embeddings della stessa dimensione
3. **Composizione** → Combinazione nodo-relazione
4. **Message Passing** → Scambio di informazioni tra vicini
5. **Aggregazione** → Combinazione dei messaggi ricevuti
6. **Ripetizione** → Processo ripetuto per più layers
7. **Scoring** → Calcolo probabilità di nuove relazioni
8. **Apprendimento** → Aggiustamento dei parametri basato sugli errori

**Obiettivo finale**: Il modello impara a rappresentare entità e relazioni in modo che possa predire accuratamente nuove connessioni nel grafo di conoscenza.