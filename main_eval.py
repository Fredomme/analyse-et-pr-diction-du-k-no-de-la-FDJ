import numpy as np
import pandas as pd
import torch
import os
import yaml
from datetime import datetime

from models import LSTMWithEmbedding, TransformerWithTime, get_device
from predictor import predict_draw, predict_draw_transformer
from metrics import evaluate_draws
from fdj_filter import FDJFilter

def bootstrap_confidence_interval(values, n_boot=1000, alpha=0.05):
    """
    Calcule l'intervalle de confiance via bootstrap percentile.
    """
    import random
    n = len(values)
    stats = []
    for _ in range(n_boot):
        sample = [random.choice(values) for __ in range(n)]
        stats.append(np.mean(sample))
    stats_sorted = sorted(stats)
    lower = stats_sorted[int((alpha/2)*n_boot)]
    upper = stats_sorted[int((1 - alpha/2)*n_boot)]
    return (np.mean(values), lower, upper)

def main():
    """
    1) Charger test_data et test_feats
    2) Charger les modèles LSTM et Transformer
    3) Faire les prédictions et calculer les métriques (Jaccard, Precision@20, Recall@20)
    4) Calculer un intervalle de confiance bootstrap et appliquer le filtre FDJ
    """
    device = get_device()

    if not os.path.exists("models_out/test_data.npy"):
        print("[ERROR] Pas de test_data.npy => exécuter d'abord main_train.")
        return

    test_data = np.load("models_out/test_data.npy")
    test_feats = pd.read_parquet("models_out/test_feats.parquet")

    # Charger configuration si disponible
    if os.path.exists("models_out/config_used.yaml"):
        with open("models_out/config_used.yaml", "r") as f:
            config_used = yaml.safe_load(f)
    else:
        config_used = {}

    # Charger le modèle LSTM
    model_lstm = LSTMWithEmbedding(
        vocab_size=71,
        embed_dim=32,
        hidden_dim=config_used.get("training", {}).get("lstm_hidden_dim", 256),
        num_layers=config_used.get("training", {}).get("lstm_num_layers", 3),
        dropout=config_used.get("training", {}).get("lstm_dropout", 0.3)
    )
    model_lstm.load_state_dict(torch.load("models_out/model_lstm.pt", map_location=device))
    model_lstm.to(device)
    model_lstm.eval()

    # Charger le modèle Transformer
    model_trans = TransformerWithTime(
        vocab_size=71,
        embed_dim=32,
        nhead=config_used.get("training", {}).get("trans_nhead", 4),
        num_layers=config_used.get("training", {}).get("trans_num_layers", 3),
        hidden_dim=config_used.get("training", {}).get("trans_hidden_dim", 256),
        time_feat_dim=config_used.get("training", {}).get("trans_time_feat_dim", 16),
        dropout=config_used.get("training", {}).get("trans_dropout", 0.3)
    )
    model_trans.load_state_dict(torch.load("models_out/model_trans.pt", map_location=device))
    model_trans.to(device)
    model_trans.eval()

    # Prédiction avec LSTM
    pred_lstm = predict_draw(model_lstm, test_data)

    # Prédiction avec Transformer : mise à jour pour utiliser 7 features temporelles
    time_cols = ["day_of_week", "day_of_month", "month", "year", "dow_sin", "dow_cos", "month_sin"]
    test_time = test_feats[time_cols].values
    pred_trans = predict_draw_transformer(model_trans, test_data, test_time)

    # Calcul des métriques (attention à l'alignement : on compare les prédictions sur test_data[1:] avec test_data[1:])
    true_test = [list(test_data[i+1]) for i in range(len(test_data)-1)]
    metrics_lstm = evaluate_draws(pred_lstm[:-1], true_test)
    metrics_trans = evaluate_draws(pred_trans, true_test)

    print("[LSTM] Jaccard =", metrics_lstm["jaccard_mean"], 
          " Precision@20 =", metrics_lstm["precision_mean"], 
          " Recall@20 =", metrics_lstm["recall_mean"])
    print("[TRANS] Jaccard =", metrics_trans["jaccard_mean"], 
          " Precision@20 =", metrics_trans["precision_mean"], 
          " Recall@20 =", metrics_trans["recall_mean"])

    # Calcul d'un intervalle de confiance bootstrap sur le score Jaccard pour LSTM
    from metrics import jaccard_similarity
    jacc_list = []
    for pr, tr in zip(pred_lstm[:-1], true_test):
        jacc_list.append(jaccard_similarity(set(pr), set(tr)))
    mean_jacc, low_jacc, high_jacc = bootstrap_confidence_interval(jacc_list, n_boot=1000, alpha=0.05)
    print(f"[LSTM] Jaccard ~ {mean_jacc:.3f} IC95% [{low_jacc:.3f}, {high_jacc:.3f}]")

    # Application du filtre FDJ sur le dernier tirage prédit par LSTM
    if os.path.exists("meta_info.txt"):
        hot, cold, top_pairs = [], [], []
        with open("meta_info.txt", "r") as fm:
            lines = fm.read().splitlines()
            for line in lines:
                if line.startswith("hot="):
                    hot = list(map(int, line.split("=")[1].split(",")))
                elif line.startswith("cold="):
                    cold = list(map(int, line.split("=")[1].split(",")))
                elif line.startswith("top_pairs="):
                    parts = line.split("=")[1].split(";")
                    for p in parts:
                        if "-" in p:
                            a, b = p.split("-")
                            top_pairs.append((int(a), int(b)))

        fdj = FDJFilter(
            hot_numbers=hot,
            cold_numbers=cold,
            suspect_nums=[56,38,16,15,69,44],
            top_pairs=top_pairs,
            suspicious_groups=[[10,16,38,44], [11,16,38], [33,38,45]],
            memory_buffer=None,
            max_boules=20
        )
        if len(pred_lstm) > 0:
            raw_draw = pred_lstm[-1]
            corrected = fdj.filter_draw(raw_draw)
            print("[FDJ] Dernier tirage LSTM => Brut:", raw_draw)
            print("[FDJ] Dernier tirage LSTM => Filtré:", corrected)

    # Log CSV
    with open("eval_log.csv", "a") as fev:
        fev.write(f"{datetime.now()},LSTM_jacc={metrics_lstm['jaccard_mean']},Trans_jacc={metrics_trans['jaccard_mean']}\n")

    print("[INFO] Fin de l'évaluation.")

if __name__ == "__main__":
    main()
