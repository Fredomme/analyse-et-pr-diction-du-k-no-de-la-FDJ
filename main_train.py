import os
import pandas as pd
import numpy as np
import torch
import random

from data_loader import transform_draws_as_sequences, extract_time_features
from predictor import train_model_lstm_embed, train_model_transformer

def main():
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Chargement des données prétraitées
    df_keno = pd.read_parquet("df_keno.parquet")
    df_roll = pd.read_parquet("df_rolling.parquet")

    draws_2d = transform_draws_as_sequences(df_keno)

    df_time = extract_time_features(df_keno)
    # Ajout des features temporelles cycliques
    df_time["dow_sin"] = np.sin(2 * np.pi * df_time["day_of_week"] / 7)
    df_time["dow_cos"] = np.cos(2 * np.pi * df_time["day_of_week"] / 7)
    df_time["month_sin"] = np.sin(2 * np.pi * df_time["month"] / 12)

    df_features = pd.concat([df_time.reset_index(drop=True), df_roll.reset_index(drop=True)], axis=1)

    ratio_train = 0.8
    N = len(draws_2d)
    cutoff = int(N * ratio_train)
    train_data, test_data = draws_2d[:cutoff], draws_2d[cutoff:]
    train_feats, test_feats = df_features.iloc[:cutoff], df_features.iloc[cutoff:]

    val_ratio = 0.1
    cutoff2 = int(len(train_data) * (1 - val_ratio))
    subtrain_data, val_data = train_data[:cutoff2], train_data[cutoff2:]
    subtrain_feats, val_feats = train_feats.iloc[:cutoff2], train_feats.iloc[cutoff2:]

    # Entraînement du modèle LSTM
    model_lstm = train_model_lstm_embed(
        train_data=subtrain_data,
        val_data=val_data,
        epochs=15,
        batch_size=32,
        lr=1e-3,
        log_csv="train_log_lstm.csv"
    )

    # Entraînement du modèle Transformer
    # Mise à jour pour utiliser 7 caractéristiques temporelles
    time_cols = ["day_of_week", "day_of_month", "month", "year", "dow_sin", "dow_cos", "month_sin"]
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

    os.makedirs("models_out", exist_ok=True)
    torch.save(model_lstm.state_dict(), "models_out/model_lstm.pt")
    torch.save(model_trans.state_dict(), "models_out/model_trans.pt")

    np.save("models_out/test_data.npy", test_data)
    test_feats.to_parquet("models_out/test_feats.parquet", index=False)

    print("[INFO] Entraînement terminé, modèles et test_data sauvegardés.")

if __name__ == "__main__":
    main()
