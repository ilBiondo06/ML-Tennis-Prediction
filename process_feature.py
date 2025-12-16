import pandas as pd
import numpy as np
from collections import deque

# ---------------------------------------------------------
# CONFIGURAZIONE FILE
# ---------------------------------------------------------
INPUT_FILE_MATCHES = "atp_matches_2000_2024_cleaned.csv"
INPUT_FILE_RANK_90 = "atp_rankings_90s.csv"
OUTPUT_FILE_ML = "dataset_ml_ready.csv"
OUTPUT_FILE_READABLE = "dataset_enriched.csv"

# ---------------------------------------------------------
# 1. FUNZIONI DI SUPPORTO
# ---------------------------------------------------------

def load_initial_rankings(rank_file):
    print(f"--- Caricamento Ranking Iniziale da {rank_file} ---")
    try:
        df_rank = pd.read_csv(rank_file)
        last_date = df_rank['ranking_date'].max()
        df_init = df_rank[df_rank['ranking_date'] == last_date].copy()
        
        initial_elo_dict = {}
        initial_matches_dict = {}
        
        for _, row in df_init.iterrows():
            pid = row['player']
            rank = row['rank']
            if rank < 1: rank = 1
            elo_val = 2500 - 300 * np.log10(rank)
            if elo_val < 1400: elo_val = 1400
            initial_elo_dict[pid] = elo_val
            
            if rank <= 10: est_matches = 500
            elif rank <= 50: est_matches = 300
            elif rank <= 100: est_matches = 150
            elif rank <= 500: est_matches = 50
            else: est_matches = 10  
            initial_matches_dict[pid] = est_matches
            
        print(f"Inizializzati {len(initial_elo_dict)} giocatori.") 
        return initial_elo_dict, initial_matches_dict
    except Exception as e:
        print(f"ATTENZIONE: Impossibile caricare ranking ({e}).")
        return {}, {}

def get_k_factor_538(matches_count, tourney_level):
    k_base = 250 / ((matches_count + 5) ** 0.4)
    if tourney_level == 'G': multiplier = 1.1
    elif tourney_level == 'M' or tourney_level == 'F': multiplier = 1.05
    else: multiplier = 1.0
    return k_base * multiplier

# ---------------------------------------------------------
# 2. CORE LOGIC: CALCOLO FEATURE COMPLESSE
# ---------------------------------------------------------

def calculate_features(df, initial_elos, initial_matches):
    print("Calcolo ELO, H2H e Feature Avanzate (Gradienti, Form)...")
    
    # --- STRUTTURE DATI ---
    elo_overall = initial_elos.copy()
    elo_surface = {'Hard': initial_elos.copy(), 'Clay': initial_elos.copy(), 
                   'Grass': initial_elos.copy(), 'Carpet': initial_elos.copy()}
    matches_played = initial_matches.copy()
    h2h_history = {} 

    # --- MEMORIA STORICA PER CALCOLI ROLLING ---
    win_history = {} 
    elo_history = {} 

    # --- LISTE OUTPUT ---
    feats = {
        'w_elo': [], 'l_elo': [], 'w_surface_elo': [], 'l_surface_elo': [],
        'w_h2h': [], 'l_h2h': [], 'w_prob_elo': [],
        'w_win_last_5': [], 'l_win_last_5': [],
        'w_win_last_25': [], 'l_win_last_25': [],
        'w_win_last_50': [], 'l_win_last_50': [],
        'w_win_last_100': [], 'l_win_last_100': [],
        'w_elo_grad_20': [], 'l_elo_grad_20': [],
        'w_elo_grad_35': [], 'l_elo_grad_35': [],
        'w_elo_grad_50': [], 'l_elo_grad_50': [],
        'w_elo_grad_100': [], 'l_elo_grad_100': [],
        'w_matches_played': [], 'l_matches_played': []
    }

    def get_win_pct(pid, n):
        if pid not in win_history: return 0.0
        history = list(win_history[pid])
        if not history: return 0.0
        last_n = history[-n:] 
        return sum(last_n) / len(last_n)

    def get_elo_gradient(pid, n):
        if pid not in elo_history: return 0.0
        hist = list(elo_history[pid])
        if len(hist) < 2: return 0.0
        current_elo = hist[-1]
        past_elo = hist[-n] if len(hist) >= n else hist[0]
        return current_elo - past_elo

    # --- CICLO PRINCIPALE ---
    for idx, row in df.iterrows():
        wid, lid = row['winner_id'], row['loser_id']
        level = row['tourney_level']
        surface = row['surface']
        
        # --- FIX BUG: Inizializzazione Unificata ---
        # Controlliamo singolarmente ogni giocatore.
        # Se manca dalla memoria storica, lo inizializziamo.
        
        # GESTIONE VINCITORE
        if wid not in elo_history:
            win_history[wid] = deque(maxlen=100)
            elo_history[wid] = deque(maxlen=101)
            
            # Se non ha nemmeno l'ELO (Nuovo Giocatore)
            if wid not in elo_overall:
                r = row['winner_rank'] if row['winner_rank'] < 2000 else 500
                elo_val = max(1400, 2500 - 300 * np.log10(r) if r > 0 else 1500)
                elo_overall[wid] = elo_val
                matches_played[wid] = 0
                for s in elo_surface: elo_surface[s][wid] = elo_val
            
            # Importante: Aggiungiamo il valore iniziale alla storia!
            elo_history[wid].append(elo_overall[wid])

        # GESTIONE PERDENTE
        if lid not in elo_history:
            win_history[lid] = deque(maxlen=100)
            elo_history[lid] = deque(maxlen=101)
            
            if lid not in elo_overall:
                r = row['loser_rank'] if row['loser_rank'] < 2000 else 500
                elo_val = max(1400, 2500 - 300 * np.log10(r) if r > 0 else 1500)
                elo_overall[lid] = elo_val
                matches_played[lid] = 0
                for s in elo_surface: elo_surface[s][lid] = elo_val
            
            elo_history[lid].append(elo_overall[lid])

        # --- SALVATAGGIO STATO ATTUALE (PRE-MATCH) ---
        curr_w_elo = elo_overall[wid]
        curr_l_elo = elo_overall[lid]
        s_clean = surface if surface in elo_surface else 'Hard'
        curr_w_surf = elo_surface[s_clean].get(wid, curr_w_elo)
        curr_l_surf = elo_surface[s_clean].get(lid, curr_l_elo)
        
        pair_key = tuple(sorted((wid, lid)))
        if pair_key not in h2h_history: h2h_history[pair_key] = {wid: 0, lid: 0}
        
        feats['w_elo'].append(curr_w_elo)
        feats['l_elo'].append(curr_l_elo)
        feats['w_surface_elo'].append(curr_w_surf)
        feats['l_surface_elo'].append(curr_l_surf)
        feats['w_h2h'].append(h2h_history[pair_key][wid])
        feats['l_h2h'].append(h2h_history[pair_key][lid])
        feats['w_matches_played'].append(matches_played[wid])
        feats['l_matches_played'].append(matches_played[lid])

        # --- CALCOLO FEATURE ROLLING ---
        windows_win = [5, 25, 50, 100]
        windows_grad = [20, 35, 50, 100]
        
        for w in windows_win:
            feats[f'w_win_last_{w}'].append(get_win_pct(wid, w))
            feats[f'l_win_last_{w}'].append(get_win_pct(lid, w))
            
        for w in windows_grad:
            feats[f'w_elo_grad_{w}'].append(get_elo_gradient(wid, w))
            feats[f'l_elo_grad_{w}'].append(get_elo_gradient(lid, w))

        # --- CALCOLO ELO POST-MATCH ---
        w_mix = 0.6 * curr_w_elo + 0.4 * curr_w_surf
        l_mix = 0.6 * curr_l_elo + 0.4 * curr_l_surf
        prob_w = 1 / (1 + 10 ** ((l_mix - w_mix) / 400))
        feats['w_prob_elo'].append(prob_w)
        
        k_w = get_k_factor_538(matches_played[wid], level)
        k_l = get_k_factor_538(matches_played[lid], level)
        delta_w = k_w * (1 - prob_w)
        delta_l = k_l * (0 - (1 - prob_w))
        
        elo_overall[wid] += delta_w
        elo_overall[lid] += delta_l
        elo_surface[s_clean][wid] += delta_w
        elo_surface[s_clean][lid] += delta_l
        matches_played[wid] += 1
        matches_played[lid] += 1
        h2h_history[pair_key][wid] += 1
        
        # Aggiornamento Storico
        win_history[wid].append(1)
        win_history[lid].append(0)
        elo_history[wid].append(elo_overall[wid])
        elo_history[lid].append(elo_overall[lid])

    for k, v in feats.items():
        df[k] = v
        
    return df

# ---------------------------------------------------------
# 3. DATASET ML (Differenze & Randomizzazione)
# ---------------------------------------------------------

def transform_to_ml_format(df):
    print("Trasformazione finale per ML (Calcolo Differenze)...")
    
    # Lista colonne base da mantenere
    cols_base = [
        'tourney_date', 'surface', 'tourney_level', 'round', 'match_num', 'best_of', # Aggiunto BEST_OF
        'winner_id', 'winner_name', 'winner_rank', 'winner_rank_points', 'winner_age', 'winner_hand', 'winner_ht', 
        'loser_id', 'loser_name', 'loser_rank', 'loser_rank_points', 'loser_age', 'loser_hand', 'loser_ht'
    ]
    
    # Aggiungiamo tutte le feature calcolate (quelle che iniziano con w_ o l_)
    # w_elo, l_elo, w_win_last_5, l_win_last_5, etc...
    calc_cols = [c for c in df.columns if c.startswith('w_') or c.startswith('l_')]
    
    df_ml = df[cols_base + calc_cols].copy()
    
    # Rinomina Winner -> p1, Loser -> p2
    rename_map = {}
    for c in df_ml.columns:
        if c.startswith('winner') or c.startswith('w_'):
            rename_map[c] = c.replace('winner', 'p1').replace('w_', 'p1_')
        elif c.startswith('loser') or c.startswith('l_'):
            rename_map[c] = c.replace('loser', 'p2').replace('l_', 'p2_')
    df_ml = df_ml.rename(columns=rename_map)
    
    df_ml['target'] = 1 
    
    # --- RANDOM SWAP ---
    mask = np.random.rand(len(df_ml)) < 0.5
    p1_cols = [c for c in df_ml.columns if 'p1' in c]
    for c1 in p1_cols:
        c2 = c1.replace('p1', 'p2')
        if c2 in df_ml.columns:
            df_ml.loc[mask, [c1, c2]] = df_ml.loc[mask, [c2, c1]].values
    df_ml.loc[mask, 'target'] = 0
    
    # --- FEATURE ENGINEERING DIFFERENZIALE (Tutte le _DIFF richieste) ---
    
    # 1. Ranking & Punti
    df_ml['ATP_RANK_DIFF'] = df_ml['p2_rank'] - df_ml['p1_rank'] # Positivo se P1 è meglio (Rank più basso)
    df_ml['ATP_POINT_DIFF'] = df_ml['p1_rank_points'] - df_ml['p2_rank_points']
    
    # 2. ELO
    df_ml['DIFF_ELO'] = df_ml['p1_elo'] - df_ml['p2_elo']
    
    # 3. Esperienza (N Games)
    df_ml['DIFF_N_GAMES'] = df_ml['p1_matches_played'] - df_ml['p2_matches_played']
    
    # 4. Win % Last X (Forma)
    for x in [5, 25, 50, 100]:
        c1, c2 = f'p1_win_last_{x}', f'p2_win_last_{x}'
        df_ml[f'WIN_LAST_{x}_DIFF'] = df_ml[c1] - df_ml[c2]
        
    # 5. ELO Gradients (Trend)
    for x in [20, 35, 50, 100]:
        c1, c2 = f'p1_elo_grad_{x}', f'p2_elo_grad_{x}'
        df_ml[f'ELO_GRAD_{x}_DIFF'] = df_ml[c1] - df_ml[c2]
        
    # 6. Altro
    df_ml['diff_age'] = df_ml['p1_age'] - df_ml['p2_age']
    df_ml['diff_ht'] = df_ml['p1_ht'] - df_ml['p2_ht']
    df_ml['p1_is_left'] = (df_ml['p1_hand'] == 'L').astype(int)
    df_ml['p2_is_left'] = (df_ml['p2_hand'] == 'L').astype(int)
    
    # Pulizia finale (BEST_OF è già presente come colonna, la rinominiamo per consistenza se vuoi)
    # df_ml['BEST_OF'] = df_ml['best_of'] 
    
    return df_ml

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("--- ELABORAZIONE FEATURES ---")
    init_elo, init_matches = load_initial_rankings(INPUT_FILE_RANK_90)
    
    try:
        df = pd.read_csv(INPUT_FILE_MATCHES)
    except FileNotFoundError:
        print(f"ERRORE: Non trovo {INPUT_FILE_MATCHES}.")
        return

    df_enriched = calculate_features(df, init_elo, init_matches)
    df_enriched.to_csv(OUTPUT_FILE_READABLE, index=False)
    print(f"File leggibile salvato: {OUTPUT_FILE_READABLE}")
    
    df_ml = transform_to_ml_format(df_enriched)
    df_ml = df_ml.fillna(0)
    
    df_ml.to_csv(OUTPUT_FILE_ML, index=False)
    print(f"Dataset ML Completo salvato: {OUTPUT_FILE_ML}")
    print("Colonne generate:", [c for c in df_ml.columns if 'DIFF' in c])

if __name__ == "__main__":
    main()