import pandas as pd
import numpy as np

# CONFIGURAZIONE
INPUT_FILE = "atp_matches_2000_2024_raw.csv"
OUTPUT_FILE = "atp_matches_2000_2024_cleaned.csv"

def clean_data(df):
    print(f"Righe originali: {len(df)}")
    
    # 1. GESTIONE TESTE DI SERIE (SEED)
    # Se vuoto, non è testa di serie,quindi mettiamo un valore alto (es. 100, signifia che è il 100esimo favorito alla vittoria del torneo)
    # così il modello capisce che vale meno della testa di serie n. 32
    df['winner_seed'] = df['winner_seed'].fillna(100)
    df['loser_seed'] = df['loser_seed'].fillna(100)
    
    # 2. GESTIONE ENTRY (WC, Q, etc.)
    # Se vuoto, è un ingresso Standard ('STD')
    df['winner_entry'] = df['winner_entry'].fillna('STD')
    df['loser_entry'] = df['loser_entry'].fillna('STD')
    
    # 3. GESTIONE RANKING
    # Se vuoto, assumiamo sia un ranking basso (es. 2000) non dovrebbero esserci player con rank o punti vuoti
    df['winner_rank'] = df['winner_rank'].fillna(2000)
    df['loser_rank'] = df['loser_rank'].fillna(2000)
    df['winner_rank_points'] = df['winner_rank_points'].fillna(0)
    df['loser_rank_points'] = df['loser_rank_points'].fillna(0)
    
    # 4. GESTIONE DATI FISICI (Altezza, Età) -> MEDIA
    # Calcoliamo la media globale per riempire gli attributi vuoti
    avg_age_w = df['winner_age'].mean()
    avg_age_l = df['loser_age'].mean()
    avg_ht_w = df['winner_ht'].mean()
    avg_ht_l = df['loser_ht'].mean()
    
    df['winner_age'] = df['winner_age'].fillna(avg_age_w)
    df['loser_age'] = df['loser_age'].fillna(avg_age_l)
    df['winner_ht'] = df['winner_ht'].fillna(avg_ht_w)
    df['loser_ht'] = df['loser_ht'].fillna(avg_ht_l)
    
    # Gestione mano (Destra/Sinistra), visto che la maggior parte è destra, riempiamo con 'R'
    df['winner_hand'] = df['winner_hand'].fillna('R')
    df['loser_hand'] = df['loser_hand'].fillna('R')
    
    # 5. GESTIONE SURFACE (Superficie)
    # Se mancano poche righe, meglio rimuoverle (non dovrebbe essere un dato mancante)
    df = df.dropna(subset=['surface']).copy()
    
    # 6. GESTIONE STATISTICHE MATCH (Ace, DF, Minuti...)
    # Riempiamo con -1 perchè questi dati non saranno input diretto (non posso sapere a priori le statistiche di un match futuro)
    # ma se calcoliamo le medie storiche dobbiamo ricordarci di toglierli controllare solo i valori >=0
    # Per 'minutes' usiamo la media di tutte le partite
    stats_cols = [
        'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    ]
    
    for col in stats_cols:
        # Se è 'minutes' usiamo la media, per gli altri (conteggi) usiamo 0
        if col == 'minutes':
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(-1)
            
    print(f"Righe dopo pulizia: {len(df)}")
    
    # Verifica finale
    missing = df.isnull().sum().sum()
    print(f"Valori mancanti residui: {missing}")
    
    return df

def main():
    print("--- INIZIO PULIZIA DATI ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"File {INPUT_FILE} non trovato. Esegui prima lo script 1!")
        return

    df_clean = clean_data(df)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset pulito salvato come: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()