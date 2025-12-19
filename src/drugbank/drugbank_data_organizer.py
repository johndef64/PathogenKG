
#%%
# Uniform drugbank data to vitagraph
import pandas as pd
import os

abs_path = os.path.abspath(__file__)
current_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
print(f"Absolute path of the current file: {abs_path}")
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# two_levels_up = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


root = os.path.join(parent_dir, 'dataset', 'DRUGBANK')  
# Impostare il percorso del file CSV

os.chdir(root)

# Caricare il dataset
file_path = os.path.join(root,r'scraped\drug_target_uniprotid.csv')  
df = pd.read_csv(file_path)
df
#%%
# Creare il dataset di triple
triples = []

for _, row in df.iterrows():
    drug_id = row['DrugBank_ID']
    targets = row['Target_UniProt_IDs']
    
    # Processare solo le righe che hanno target validi (non "NOT_FOUND")
    if targets != 'NOT_FOUND':
        target_list = targets.split()  # Dividere i target multipli separati da spazi
        
        # Creare una tripla per ogni target
        for target in target_list:
            triple = {
                'head': f'Compound::DrugBank:{drug_id}',
                'interaction': 'TARGET',
                'tail': f'Gene::Uniprot:{target}',
                'source': 'DRUGBANK',
                'type': 'Compound-Gene'
            }
            triples.append(triple)

# Convertire in DataFrame
triples_df = pd.DataFrame(triples)

# Visualizzare il risultato
print(triples_df.head())

# Salvare il dataset di triple (opzionale)
triples_df.to_csv('vitagraph_drug_target_triples.csv', index=False)


# %%
os.getcwd()
# %%
import os
import pandas as pd

# Percorsi forniti
file_path1 = r'G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\DRUGBANK\drugbank_uniprot_to_taxonomy.csv'
file_path2 = r'G:\Altri computer\Horizon\horizon_workspace\projects\DatabaseRetrieval\KnowledgeGraphs\VitaExt\dataset\DRUGBANK\drugbank_to_uniprot_mapping.csv'

try:
    # Carica i file (sostituisci con i percorsi corretti)
    df_taxonomy = pd.read_csv(file_path1)
    df_uniprot = pd.read_csv(file_path2)
    
    print("Dataset 1 (taxonomy):")
    print(df_taxonomy.head())
    print(f"Colonne: {df_taxonomy.columns.tolist()}")
    
    print("\nDataset 2 (uniprot mapping):")
    print(df_uniprot.head())
    print(f"Colonne: {df_uniprot.columns.tolist()}")
    
    # Merge dei dataset
    # Assumendo che le colonne si chiamino 'uniprot_id' e 'UniProt_ID'
    merged_df = pd.merge(df_uniprot, df_taxonomy, 
                        left_on='UniProt_ID', right_on='uniprot_id', 
                        how='left')
    
    # Rimuovi colonna duplicata se necessario
    if 'uniprot_id' in merged_df.columns:
        merged_df = merged_df.drop('uniprot_id', axis=1)
    
    print(f"\nDataset unificato - {len(merged_df)} righe:")
    print(merged_df.head())
    
    # Salva il risultato
    output_path = 'drugbank_unprot_taxonomy.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\nDataset salvato in: {output_path}")
    
except FileNotFoundError as e:
    print(f"Errore: File non trovato - {e}")
except Exception as e:
    print(f"Errore generico: {e}")

# %%