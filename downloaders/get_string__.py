import os
import pandas as pd
import glob

# filefolder = os.path.dirname(os.path.abspath(__file__))
os.chdir('../')
os.getcwd()
#%%
os.chdir('dataset')
enseml_genes = pd.read_csv('ensembl_data.zip')
enseml_genes
#%%
datafolder = os.path.join('STRING')

os.chdir(datafolder)

#%%
# gallus gallus 9031 , canis lupus 9615
human = 9606
pathogens =[83332, 224308, 208964, 99287, 71421, 243230, 
            85962, 171101, 243277, 294, 1314, 272631,
            212717, 36329, 237561, 6183, 5664, 185431, 330879]


import requests
import os

# Lista degli ID di taxonomia che vogliamo scaricare
tax_ids = ['83332', '294']  # ad esempio, 83332 = Homo sapiens, 294 = Saccharomyces cerevisiae

# URL base per il download
base_url = 'https://stringdb-downloads.org/download/'

# Lista dei file da scaricare per ogni tax_id
files = [
    ('protein.links.v12.0', 'protein.links.v12.0'),
    ('protein.physical.links.v12.0', 'protein.physical.links.v12.0'),
    ('protein.aliases.v12.0', 'protein.aliases.v12.0'),
    ('protein.enrichment.terms.v12.0', 'protein.enrichment.terms.v12.0'),
    ('protein.orthology.v12.0', 'protein.orthology.v12.0')
]

def download_string_files(tax_id, suffix, subdir, output_folder):
    """Scarica e salva un file da STRING database in una cartella specifica."""
    url = f"{base_url}{subdir}/{tax_id}.{suffix}.txt.gz"

    try:
        # Crea la cartella di output se non esiste
        os.makedirs(output_folder, exist_ok=True)

        # Costruisci il nome del file di output
        filename = f"{tax_id}.{suffix}.txt.gz"
        output_path = os.path.join(output_folder, filename)

        # Invia una richiesta HTTP con un user-agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Solleva un'eccezione per codici di errore HTTP

        # Scrivi il contenuto della risposta nel file
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"File scaricato e salvato in: {output_path}")
        return True

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP durante il download di {url}: {http_err}")
    except Exception as err:
        print(f"Errore generico durante il download di {url}: {err}")
    return False

# Specifica la cartella in cui vuoi salvare i file (ad esempio, "string_files")
output_folder = "string_files"

# Esegui il download per ogni tax_id e ogni file
for tax_id in tax_ids:
    print(f"\nInizio download per tax_id: {tax_id}")
    for suffix, subdir in files:
        download_string_files(tax_id, suffix, subdir, output_folder)
    print(f"Download completati per tax_id: {tax_id}\n")

print(f"\nTutti i file sono stati salvati nella cartella: {os.path.abspath(output_folder)}")



#%%


import requests
import gzip
from io import BytesIO

def ScaricaElaboraFile(tax_id):
    """
    Scarica ed elabora il file per un determinato tax_id.
    Restuisce un dizionario con le informazioni se il download va a buon fine, altrimenti None.
    """
    url = f"https://stringdb-static.org/download/protein.actions.v11.0/{tax_id}.protein.actions.v11.0.txt.gz"

    try:
        # Esegue la richiesta HTTP per scaricare il file
        response = requests.get(url, timeout=10)

        # Verifica se il download è andato a buon fine
        if response.status_code == 200:
            # Crea un oggetto BytesIO dal contenuto della risposta
            data = BytesIO(response.content)

            # Apre il file gzippato in modalità lettura
            with gzip.open(data, 'rt') as f:
                # Legge le righe del file
                lines = f.readlines()

                # Verifica se il file è vuoto
                if not lines:
                    print(f"File per tax_id {tax_id} è vuoto o non esiste.")
                    return None

                # Esempio di parsing: estrazione delle prime 5 righe
                # Puoi personalizzare qui il parsing in base alle tue necessità
                informazioni = {
                    "tax_id": tax_id,
                    "righe": lines[:5]
                }
                return informazioni

        else:
            print(f"Errore nel download del file per tax_id {tax_id}. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta per tax_id {tax_id}: {str(e)}")
        return None

# Dizionario per memorizzare i risultati
risultati = {}

for tax_id in tax_ids:
    print(f"\nElaborazione tax_id: {tax_id}")
    result = ScaricaElaboraFile(tax_id)

    if result:
        risultati[tax_id] = result
        print(f"File per tax_id {tax_id} elaborato correttamente.")
        # Stampa le prime righe come esempio
        print("Esempio delle prime righe:")
        for riga in result["righe"]:
            print(riga.strip())
    else:
        print(f"Errore nell'elaborazione del file per tax_id {tax_id}")

print("\nRisultati finali:")
for tax_id, info in risultati.items():
    print(f"Tax_id: {tax_id}")
    print(f"Numero righe prese in esempio: {len(info['righe'])}")