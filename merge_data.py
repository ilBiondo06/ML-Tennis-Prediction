import pandas as pd
import glob
import os

# CONFIGURAZIONE
INPUT_FOLDER = "datasets"
OUTPUT_FILENAME = "atp_matches_2000_2024_raw.csv"

def main():
    print("--- UNIONE DATI ---")
    
    # Cerca file che iniziano con "atp_matches_" dentro la cartella datasets
    pattern = os.path.join(INPUT_FOLDER, "atp_matches_*.csv")
    all_files = glob.glob(pattern)
    

    li = []
    for filename in all_files:
        try:
            
            df_temp = pd.read_csv(filename, index_col=None, header=0)
            
            # Convertiamo subito la data (formato YYYYMMDD) in oggetto datetime
            df_temp['tourney_date'] = pd.to_datetime(df_temp['tourney_date'], format='%Y%m%d', errors='coerce')
            
            li.append(df_temp)
        except Exception as e:
            print(f"Errore caricamento {filename}: {e}")

    if not li:
        print("Errore: Nessun file caricato.")
        return

    # Unione
    df_total = pd.concat(li, axis=0, ignore_index=True)
    
    # Ordinamento CRONOLOGICO (Fondamentale per l'ELO, anche se dovrebbero già essere ordinati)
    df_total = df_total.sort_values(by=['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)

    #Salvo il dataset intero
    df_total.to_csv(OUTPUT_FILENAME, index=False)
    print(f"--- COMPLETATO ---")
    print(f"File unito salvato come: {OUTPUT_FILENAME}")
    print(f"Totale partite: {len(df_total)}")
    print(df_total[['tourney_date', 'winner_name', 'loser_name']].head())

if __name__ == "__main__":
    main()