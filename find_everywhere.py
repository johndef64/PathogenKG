import os
import sys
from tqdm import tqdm # Importiamo la libreria tqdm per la barra di avanzamento

# # Solo ricerca base (default 2 MB)
# python .\find_everywhere.py "full_uri"

# # Con limite di dimensione personalizzato
# python .\find_everywhere.py "full_uri" 5

# # Escludendo file .csv
# python .\find_everywhere.py "full_uri" 5 .csv

# # Escludendo più estensioni (separate da virgola)
# python .\find_everywhere.py "full_uri" 5 .csv,.log,.tmp

# # Anche senza il punto funziona
# python .\find_everywhere.py "full_uri" 5 csv,log,tmp


def cerca_stringa_nei_file(stringa_cercata, max_size_mb=2, exclude_extensions=None):
    """
    Cerca una stringa specificata in tutti i file della directory corrente 
    mostrando una barra di avanzamento.
    
    Args:
        stringa_cercata (str): La stringa di testo da cercare.
        max_size_mb (float): Dimensione massima dei file da scansionare in MB (default: 2).
        exclude_extensions (list): Lista di estensioni da escludere (es. ['.log', '.tmp']).
    """
    if exclude_extensions is None:
        exclude_extensions = []
    directory_corrente = os.getcwd() 
    
    print(f"Sto cercando la stringa: '**{stringa_cercata}**' nella cartella: {directory_corrente}")
    print(f"Dimensione massima file: {max_size_mb} MB")
    if exclude_extensions:
        print(f"Estensioni escluse: {', '.join(exclude_extensions)}")
    print()
    
    # Prepara la lista di tutti i file da scansionare, escludendo cartelle e lo script stesso
    file_da_scansionare = []
    file_saltati_dimensione = 0
    file_saltati_estensione = 0
    nome_script = os.path.basename(sys.argv[0])
    max_size_bytes = max_size_mb * 1024 * 1024  # Converti MB in bytes

    # Scansiona ricorsivamente tutti i file in tutte le sottocartelle
    for root, dirs, files in os.walk(directory_corrente):
        for nome in files:
            percorso_completo = os.path.join(root, nome)
            if nome != nome_script:
                # Controlla l'estensione
                _, estensione = os.path.splitext(nome)
                if estensione.lower() in [ext.lower() for ext in exclude_extensions]:
                    file_saltati_estensione += 1
                    continue
                    
                try:
                    # Controlla la dimensione del file
                    dimensione_file = os.path.getsize(percorso_completo)
                    if dimensione_file <= max_size_bytes:
                        file_da_scansionare.append(percorso_completo)
                    else:
                        file_saltati_dimensione += 1
                except Exception:
                    # Se non riesci a ottenere la dimensione, salta il file
                    file_saltati_dimensione += 1
    
    if file_saltati_dimensione > 0:
        print(f"⚠️ {file_saltati_dimensione} file(s) saltati perché troppo grandi (> {max_size_mb} MB)")
    if file_saltati_estensione > 0:
        print(f"⚠️ {file_saltati_estensione} file(s) saltati per estensione esclusa")
    if file_saltati_dimensione > 0 or file_saltati_estensione > 0:
        print()

    if not file_da_scansionare:
        print("❌ Nessun file testuale trovato nella directory corrente da scansionare.")
        return

    trovato_almeno_una_corrispondenza = False
    file_con_corrispondenze = {}  # Dizionario per tracciare file e numero di occorrenze
    
    # Usiamo tqdm per avvolgere la lista di file, creando una barra di avanzamento
    # La barra mostrerà il progresso man mano che ogni file viene processato
    for percorso_completo in tqdm(file_da_scansionare, desc="Scansione file"):
        
        # Inizializza il contatore di riga
        numero_riga = 0
        
        try:
            # Apriamo il file in modalità lettura ('r')
            with open(percorso_completo, 'r', encoding='utf-8', errors='ignore') as file:
                
                # Scorriamo il file riga per riga
                for riga in file:
                    numero_riga += 1
                    
                    # Verifichiamo la presenza della stringa (ignorando maiuscole/minuscole)
                    if stringa_cercata.lower() in riga.lower():
                        
                        # Restituiamo il risultato (percorso relativo per una visualizzazione più chiara)
                        percorso_relativo = os.path.relpath(percorso_completo, directory_corrente)
                        print(f"\n✔️ Trovata in File: **{percorso_relativo}**, Riga: **{numero_riga}**")
                        trovato_almeno_una_corrispondenza = True
                        
                        # Aggiorna il dizionario delle corrispondenze
                        if percorso_relativo not in file_con_corrispondenze:
                            file_con_corrispondenze[percorso_relativo] = 0
                        file_con_corrispondenze[percorso_relativo] += 1
                        
        except Exception as e:
            # Gestisce eventuali errori
            percorso_relativo = os.path.relpath(percorso_completo, directory_corrente)
            print(f"\n⚠️ Errore durante la lettura del file {percorso_relativo}: {e}")

    if not trovato_almeno_una_corrispondenza:
        print("\n---")
        print("❌ Nessuna corrispondenza trovata in nessun file.")
    else:
        # Stampa l'overview finale
        print("\n" + "="*80)
        print("📊 OVERVIEW - File con corrispondenze:")
        print("="*80)
        for file, count in sorted(file_con_corrispondenze.items()):
            print(f"  • {file} ({count} occorrenza/e)")
        print("="*80)
        print(f"Totale file con corrispondenze: {len(file_con_corrispondenze)}")
        print(f"Totale occorrenze trovate: {sum(file_con_corrispondenze.values())}")
        print("="*80)

# --- Punto di ingresso dello script ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Errore: Devi fornire la stringa da cercare come argomento!")
        print("Utilizzo corretto: python find_everywhere.py <stringa_da_cercare> [dimensione_max_MB] [estensioni_escluse]")
        print("Esempio: python find_everywhere.py 'testo' 5 .log,.tmp,.bak")
        print("Esempio: python find_everywhere.py 'testo' 2 .csv")
        sys.exit(1)
    
    # Prende la stringa dall'argomento riga di comando
    stringa_utente = sys.argv[1]
    
    # Prende la dimensione massima opzionale (default 2 MB)
    max_size = 2.0
    if len(sys.argv) >= 3:
        try:
            max_size = float(sys.argv[2])
        except ValueError:
            print(f"⚠️ Attenzione: '{sys.argv[2]}' non è un numero valido. Uso il valore default di 2 MB.")
    
    # Prende le estensioni da escludere (opzionale)
    exclude_exts = []
    if len(sys.argv) >= 4:
        # Divide per virgola e aggiunge il punto se mancante
        ext_input = sys.argv[3]
        for ext in ext_input.split(','):
            ext = ext.strip()
            if ext and not ext.startswith('.'):
                ext = '.' + ext
            if ext:
                exclude_exts.append(ext)
    
    # Avvia la funzione principale
    cerca_stringa_nei_file(stringa_utente, max_size, exclude_exts)

    # stampa alla fine un overview dei file in cui è stata trovata la stringa cercata

