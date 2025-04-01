import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

from data_loader import (
    load_keno_data,
    analyze_keno_data,
    load_final_rolling_features
)

def main():
    """
    1) Charge la configuration
    2) Charge et analyse les données Keno
    3) Calcule les rolling features et les enregistre
    4) Écrit un log CSV avec quelques métadonnées
    """
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df_keno = load_keno_data()
    hot, cold, top_pairs, keno_all = analyze_keno_data(df_keno)
    print(f"[INFO] Nombre total tirages: {len(df_keno)}")

    # Calcul des rolling features
    rolling_path = load_final_rolling_features(
        df_keno,
        windows=[20,50,100],
        chunk_size=2000,
        out_prefix="rolling_feats"
    )
    print(f"[INFO] Rolling OK => {rolling_path}")

    # Enregistrement des métadonnées
    hot_str = ",".join(str(x) for x in hot)
    cold_str = ",".join(str(x) for x in cold)
    tp_str   = ";".join(f"{a}-{b}" for (a,b) in top_pairs)
    with open("meta_info.txt","w") as fm:
        fm.write(f"hot={hot_str}\n")
        fm.write(f"cold={cold_str}\n")
        fm.write(f"top_pairs={tp_str}\n")
        fm.write(f"len_keno_all={len(keno_all.values.flatten())}\n")

    # Sauvegarde du DataFrame brut
    df_keno.to_parquet("df_keno.parquet")
    print("[INFO] df_keno.parquet sauvé.")

    # Log CSV
    with open("preproc_log.csv","a") as flog:
        flog.write(f"{datetime.now()},n_tirages={len(df_keno)},rolling_file={rolling_path}\n")

if __name__ == "__main__":
    main()
