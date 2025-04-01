"""
Pipeline amélioré pour un projet Keno à visée scientifique
---------------------------------------------------------
Voici un exemple structuré en trois scripts principaux :
1) main_preproc.py  => Prétraitement, calcul des rolling features, stockage.
2) main_train.py    => Création d’un split train/val/test (ou cross validation), entraînement des modèles.
3) main_eval.py     => Chargement des modèles et évaluation finale (avec ensembling, filtrage FDJ, etc.).

Les modules annexes (data_loader.py, fdj_filter.py, metrics.py, models.py, predictor.py)
restent globalement similaires, avec quelques ajouts mineurs pour la robustesse.

Parmi les ajouts et améliorations notables :
- Encodage temporel cyclique (jour/semaine, jour/mois, etc.).
- Exemple d’utilisation de la validation croisée glissante.
- Ajout d’une routine d’early stopping.
- Logging plus complet (CSV) + sauvegarde de la configuration hyperparamètres.

Ci-dessous, un canevas complet, commenté.
"""

###############################################
# main_preproc.py
###############################################

import os
import pandas as pd
import numpy as np

from data_loader import load_keno_data, analyze_keno_data

# Imaginons qu'on ait le module load_final_rolling_features déjà existant ou on le recode ci-dessous.

#############################
# Exemple de calcul Rolling
#############################
def compute_rolling_distributions(df, windows=[20, 50, 100]):
    """
    Calcule les fréquences de chaque boule (1..70) sur diverses fenêtres glissantes.
    Retourne un dict {w: DataFrame} où w est la taille de fenêtre.
    """
    # Suppose qu'on a un DataFrame df avec colonnes boule1..boule20.
    # On va itérer sur la colonne index.
    results = {}
    boule_cols = [f"boule{i}" for i in range(1,21)]
    arr = df[boule_cols].values
    n = len(arr)

    # Convertir en codes 1..70
    for w in windows:
        # Pour chaque taille de fenêtre, on va créer un DataFrame de shape (n, 70)
        # row i => distribution normalisée sur la fenêtre [i-w+1, i] si i-w+1 >= 0
        roll_data = np.zeros((n, 70), dtype=np.float32)
        counts_window = np.zeros(70, dtype=np.int32)
        start = 0

        # Initialiser la toute première fenêtre
        for i in range(n):
            # Ajouter la ligne i dans le buffer
            for val in arr[i]:
                if 1 <= val <= 70:
                    counts_window[val - 1] += 1

            if i - start + 1 > w:
                # Retirer la ligne start du buffer
                for val in arr[start]:
                    if 1 <= val <= 70:
                        counts_window[val - 1] -= 1
                start += 1

            # Sauvegarde la distribution
            total = (i - start + 1) * 20  # nombre de boules dans la fenêtre
            if total > 0:
                roll_data[i] = counts_window / total
            else:
                roll_data[i] = 0

        roll_df = pd.DataFrame(roll_data, columns=[f"freq_b{b}" for b in range(1,71)])
        results[w] = roll_df

    return results


def merge_rolling_features(roll_dict):
    """
    Concatène horizontalement les DataFrames de roll_dict (windows={20,50,100}).
    Retourne un unique DataFrame.
    """
    # On va faire un merge sur les index.
    # Par ex. roll_dict = {20: df_20, 50: df_50, 100: df_100}
    # On concat sur axis=1.
    dfs = []
    for w, df_w in roll_dict.items():
        renamed_cols = {col: col + f"_r{w}" for col in df_w.columns}
        df_w_renamed = df_w.rename(columns=renamed_cols)
        dfs.append(df_w_renamed)
    df_merged = pd.concat(dfs, axis=1)
    return df_merged


def main():
    # 1) Charger le dataset brut
    df_keno = load_keno_data()
    hot, cold, top_pairs, keno_all = analyze_keno_data(df_keno)
    print(f"[INFO] Nombre de tirages: {len(df_keno)}")

    # 2) Calculer Rolling Features (ou autre features)
    roll_dict = compute_rolling_distributions(df_keno, windows=[20, 50, 100])
    df_roll = merge_rolling_features(roll_dict)

    # 3) Sauvegarder :
    #   - dataset complet df_keno.parquet
    #   - df_roll.parquet
    #   - meta_info (hot, cold, etc.)

    df_keno.to_parquet("df_keno.parquet", index=False)
    df_roll.to_parquet("df_rolling.parquet", index=False)

    # Sauvegarder meta
    with open("meta_info.txt", "w") as f:
        f.write("hot=" + ",".join(map(str, hot)) + "\n")
        f.write("cold=" + ",".join(map(str, cold)) + "\n")
        f.write("top_pairs=" + ";".join(f"{a}-{b}" for (a,b) in top_pairs) + "\n")

    print("[INFO] Prétraitement terminé. Fichiers générés.")

if __name__ == "__main__":
    main()

###############################################
# main_train.py
###############################################
import numpy as np
import pandas as pd
import torch
import random

from data_loader import transform_draws_as_sequences, extract_time_features
from models import LSTMWithEmbedding, TransformerWithTime, get_device
from predictor import train_model_lstm_embed, train_model_transformer

import os

########################
# Cross Validation util
########################
def time_based_split_cv(X, n_folds=3):
    """
    Exemple simplifié de création de splits temporels.
    Divise chronologiquement X en n_folds segments successifs.
    """
    folds = []
    fold_size = len(X) // n_folds
    for i in range(n_folds):
        start_val = i * fold_size
        end_val = (i+1) * fold_size if i < n_folds - 1 else len(X)
        # train => [0 .. start_val-1], val => [start_val .. end_val-1]
        train_idx = list(range(0, start_val))
        val_idx = list(range(start_val, end_val))
        folds.append((train_idx, val_idx))
    return folds

# Optionnel: on peut coder un early_stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def main():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Charger df_keno + df_rolling
    df_keno = pd.read_parquet("df_keno.parquet")
    df_roll = pd.read_parquet("df_rolling.parquet")

    # 2) Extraire draws_2d
    draws_2d = transform_draws_as_sequences(df_keno)  # shape (N,20)

    # 3) Extraire ou créer time features
    df_time = extract_time_features(df_keno)
    # Encodage cyclique: on peut le faire direct
    # Features temporelles cycliques
    df_time["dow_sin"] = np.sin(2 * np.pi * df_time["day_of_week"] / 7)
    df_time["dow_cos"] = np.cos(2 * np.pi * df_time["day_of_week"] / 7)
    df_time["month_sin"] = np.sin(2 * np.pi * df_time["month"] / 12)

    # 4) Merger df_time + df_roll => features MLP ou Transformer
    #   (optionnel, suivant archi)
    df_features = pd.concat([df_time.reset_index(drop=True), df_roll.reset_index(drop=True)], axis=1)

    # 5) Demo: un unique split (train=80%, test=20%) ou multiple folds
    #   => on illustre un unique split + (train/val=90/10) par ex.

    ratio_train = 0.8
    N = len(draws_2d)
    cutoff = int(N * ratio_train)
    train_data = draws_2d[:cutoff]
    test_data = draws_2d[cutoff:]

    train_feats = df_features.iloc[:cutoff].reset_index(drop=True)
    test_feats = df_features.iloc[cutoff:].reset_index(drop=True)

    # 6) Sous-split train => (subtrain, val)
    val_ratio = 0.1
    cutoff2 = int(len(train_data)*(1 - val_ratio))
    subtrain_data = train_data[:cutoff2]
    val_data = train_data[cutoff2:]

    subtrain_feats = train_feats.iloc[:cutoff2]
    val_feats = train_feats.iloc[cutoff2:]

    # 7) Entraînement LSTM => ex.
    model_lstm = train_model_lstm_embed(
        train_data=subtrain_data,
        val_data=val_data,  # on passe le val_data si on veut un feedback
        epochs=15,
        batch_size=32,
        lr=1e-3,
        log_csv="train_log_lstm.csv"
    )

    # 8) Entraînement Transformer => ex.
    #  il faut transformer subtrain_feats en np pour injection time
    #  on se limite par ex. à day_of_week, day_of_month, month, year (ou dow_sin, etc.)
    time_cols = ["day_of_week", "day_of_month", "month", "year", "dow_sin", "dow_cos", "month_sin"]  # ajustez selon vos colonnes

    train_time_vals = subtrain_feats[time_cols].values
    val_time_vals = val_feats[time_cols].values

    model_trans = train_model_transformer(
        train_data=subtrain_data,
        train_time=train_time_vals,
        val_data=val_data,
        val_time=val_time_vals,
        epochs=15,
        batch_size=32,
        lr=1e-3,
        log_csv="train_log_transformer.csv"
    )

    # 9) Sauvegarde des modèles
    os.makedirs("models_out", exist_ok=True)
    torch.save(model_lstm.state_dict(), "models_out/model_lstm.pt")
    torch.save(model_trans.state_dict(), "models_out/model_trans.pt")

    # 10) Sauver test_data, test_feats
    np.save("models_out/test_data.npy", test_data)
    test_feats.to_parquet("models_out/test_feats.parquet", index=False)

    print("[INFO] Entraînement terminé, modèles et test_data sauvegardés.")

if __name__ == "__main__":
    main()

###############################################
# main_eval.py
###############################################
import numpy as np
import pandas as pd
import torch

from data_loader import transform_draws_as_sequences
from metrics import evaluate_draws
from models import LSTMWithEmbedding, TransformerWithTime, get_device
from predictor import predict_draw, predict_draw_transformer
from fdj_filter import FDJFilter

import os

def main():
    device = get_device()

    # 1) Charger test_data & test_feats
    test_data = np.load("models_out/test_data.npy")  # shape (N_test, 20)
    test_feats = pd.read_parquet("models_out/test_feats.parquet")

    # 2) Charger LSTM
    model_lstm = LSTMWithEmbedding()
    model_lstm.load_state_dict(torch.load("models_out/model_lstm.pt", map_location=device))
    model_lstm.to(device)
    model_lstm.eval()

    # 3) Charger Transformer
    model_trans = TransformerWithTime()
    model_trans.load_state_dict(torch.load("models_out/model_trans.pt", map_location=device))
    model_trans.to(device)
    model_trans.eval()

    # 4) Prédictions (LSTM, Transformer)
    # LSTM => draws
    pred_lstm = predict_draw(model_lstm, test_data)
    # Transformer => draws
    # => on a besoin time_feats pour test, ex. time_cols = [day_of_week, day_of_month, ...]
    # on fait predict_draw_transformer(model_trans, test_data, test_time)
    time_cols = ["day_of_week", "day_of_month", "month", "year"]
    test_time = test_feats[time_cols].values
    pred_trans = []
    for i in range(len(test_data)-1):
        # doc usage => (model, draws_2d, time_feats)
        # On peut manuellement coder pour un batch:
        import torch
        x_seq = torch.tensor(test_data[i], dtype=torch.long).unsqueeze(0).to(device)
        tf = torch.tensor(test_time[i+1], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model_trans(x_seq, tf)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        top_indices = probs.argsort()[-20:]
        pred = sorted(idx+1 for idx in top_indices)
        pred_trans.append(pred)

    true_test = [list(test_data[i+1]) for i in range(len(test_data)-1)]

    # 5) Calcul métriques => LSTM
    metrics_lstm = evaluate_draws(pred_lstm[:-1], true_test)  # attention alignement
    print("[LSTM] Jaccard=", metrics_lstm["jaccard_mean"])
    print("[LSTM] Precision@20=", metrics_lstm["precision_mean"])
    print("[LSTM] Recall@20=", metrics_lstm["recall_mean"])

    # Transformer
    metrics_trans = evaluate_draws(pred_trans, true_test)
    print("[TRANS] Jaccard=", metrics_trans["jaccard_mean"])
    print("[TRANS] Precision@20=", metrics_trans["precision_mean"])
    print("[TRANS] Recall@20=", metrics_trans["recall_mean"])

    # 6) Ensembling (ex: 0.4 LSTM / 0.6 TRANS) => si on veut
    # Cf. code plus poussé dans l'ancienne version

    # 7) Application Filtre FDJ sur un tirage => ex. le dernier
    with open("meta_info.txt", "r") as f:
        lines = f.read().splitlines()
        # parse hot, cold, top_pairs etc.
        hot = []
        cold = []
        top_pairs = []
        for line in lines:
            if line.startswith("hot="):
                hot = list(map(int, line.split("=")[1].split(",")))
            if line.startswith("cold="):
                cold = list(map(int, line.split("=")[1].split(",")))
            if line.startswith("top_pairs="):
                pairs_str = line.split("=")[1].split(";")
                for pstr in pairs_str:
                    if "-" in pstr:
                        a,b = pstr.split("-")
                        top_pairs.append((int(a), int(b)))

    # On récupère un tirage de test => par ex. pred_lstm[-1]
    if len(pred_lstm) > 0:
        from fdj_filter import FDJFilter
        fdj = FDJFilter(
            hot_numbers=hot,
            cold_numbers=cold,
            suspect_nums=[56,38,16,15,69,44],
            top_pairs=top_pairs,
            suspicious_groups=[[10,16,38,44],[11,16,38],[33,38,45]],
            memory_buffer=None,
            max_boules=20
        )
        raw_draw = pred_lstm[-1]
        corrected = fdj.filter_draw(raw_draw)
        print("[FDJ Filter] Tirage brut:", raw_draw)
        print("[FDJ Filter] Tirage corrigé:", corrected)

    print("[INFO] Evaluation terminée.")

if __name__ == "__main__":
    main()
