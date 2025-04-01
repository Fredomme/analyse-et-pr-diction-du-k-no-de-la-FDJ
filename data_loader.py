# data_loader.py

import os
import zipfile
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------------------------------------
# VOS CHEMINS ZIP ET EXTRACT_DIR
# -------------------------------------------------------------
zip_files = [
    "/home/fred/Téléchargements/keno.zip",
    "/home/fred/Téléchargements/keno_201811.zip",
    "/home/fred/Téléchargements/keno_202010.zip",
    "/home/fred/Téléchargements/keno_gagnant_a_vie.zip"
]
extract_dir = "/home/fred/kéno 2/keno_extracted"
os.makedirs(extract_dir, exist_ok=True)

usecols_main = ["date_de_tirage", "heure_de_tirage"] + [f"boule{i}" for i in range(1,21)]
dtype_dict   = {col: "Int64" for col in [f"boule{i}" for i in range(1,21)]}

def load_keno_data():
    """
    Extrait/charge le Keno DataFrame complet, trié par date_de_tirage.
    """
    for z in zip_files:
        with zipfile.ZipFile(z, "r") as ref:
            ref.extractall(extract_dir)

    files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".csv")]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", encoding="utf-8", low_memory=False,
                             usecols=usecols_main, dtype=dtype_dict)
            df["date_de_tirage"] = pd.to_datetime(df["date_de_tirage"], format="%d/%m/%Y", errors="coerce")
            df = df.dropna(subset=["date_de_tirage"])
            for bcol in [f"boule{i}" for i in range(1,21)]:
                df[bcol] = df[bcol].astype("Int64").fillna(-1)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Erreur lecture {f}: {e}")

    if not dfs:
        print("[ERROR] Aucune data chargée.")
        return pd.DataFrame()

    df_merged = pd.concat(dfs, ignore_index=True).dropna(subset=["date_de_tirage"])
    df_merged = df_merged.sort_values("date_de_tirage").reset_index(drop=True)
    return df_merged


def analyze_keno_data(df_keno):
    """
    Renvoie (hot, cold, top_pairs, keno_all).
    """
    boule_cols = [f"boule{i}" for i in range(1,21)]
    keno_all   = df_keno[boule_cols].astype(int)
    flat       = keno_all.values.flatten()

    hot  = pd.Series(flat).value_counts().head(10).index.tolist()
    cold = pd.Series(flat).value_counts().tail(10).index.tolist()

    cooc = Counter()
    for row in keno_all.values:
        row_sorted = sorted(row)
        for i in range(20):
            for j in range(i+1, 20):
                a, b = row_sorted[i], row_sorted[j]
                cooc[(a, b)] += 1
    top_pairs = [pair for pair,_ in cooc.most_common(50)]
    return hot, cold, top_pairs, keno_all


def transform_draws_as_sequences(df_keno):
    """
    Convertit df_keno en np.array shape (N,20).
    """
    boule_cols = [f"boule{i}" for i in range(1,21)]
    arr = df_keno[boule_cols].astype(int).values
    return arr


def extract_time_features(df_keno):
    df_time = pd.DataFrame()
    df_time["day_of_week"]  = df_keno["date_de_tirage"].dt.weekday
    df_time["day_of_month"] = df_keno["date_de_tirage"].dt.day
    df_time["month"]        = df_keno["date_de_tirage"].dt.month
    df_time["year"]         = df_keno["date_de_tirage"].dt.year
    return df_time


########################################################################
# ROLLING WINDOWS (OUT-OF-CORE)
########################################################################
def compute_rolling_one_window(df_keno, w, chunk_size=2000, out_parquet="rolling_feats.parquet"):
    """
    Calcule la rolling window `w` par chunks successifs, écrit .parquet unique.
    """
    boule_cols = [f"boule{i}" for i in range(1,21)]
    df_keno[boule_cols] = df_keno[boule_cols].astype(int)
    arr = df_keno[boule_cols].values
    n   = len(arr)

    leftover = np.zeros((0, 20), dtype=int)
    all_chunks_paths = []
    base_dir = os.path.dirname(out_parquet) if os.path.dirname(out_parquet) else "."

    os.makedirs(base_dir, exist_ok=True)
    chunk_index = 0

    import pandas as pd
    while True:
        start_idx = chunk_index * chunk_size
        if start_idx >= n:
            break
        end_idx = min(start_idx+chunk_size, n)
        block   = arr[start_idx:end_idx]

        combined = np.concatenate([leftover, block], axis=0)
        block_feats = []
        for i in range(len(leftover), len(combined)):
            start_win = max(0, i-w+1)
            window_data = combined[start_win:i+1].flatten()
            counts = pd.Series(window_data).value_counts(normalize=True)
            row_feat= [counts.get(b,0) for b in range(1,71)]
            block_feats.append(row_feat)

        leftover_len = min(w-1, len(combined))
        leftover = combined[-leftover_len:]

        col_names = [f"b_{i}_r{w}" for i in range(1,71)]
        df_block  = pd.DataFrame(block_feats, columns=col_names)

        tmp_file = os.path.join(base_dir, f"rolling_tmp_w{w}_{chunk_index}.parquet")
        df_block.to_parquet(tmp_file)
        all_chunks_paths.append(tmp_file)
        chunk_index+=1

    # Regroupement final
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer=None
    for cpath in all_chunks_paths:
        df_part = pd.read_parquet(cpath)
        table_part = pa.Table.from_pandas(df_part)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table_part.schema)
        writer.write_table(table_part)
    if writer:
        writer.close()
    # On supprime les tmp
    for cpath in all_chunks_paths:
        os.remove(cpath)

    print(f"[OK] Rolling w={w} => {out_parquet}")

def merge_parquets_out_of_core(parquet_list, out_parquet, chunk_size=2000):
    """
    Fusion horizontale (col) de plusieurs .parquet (70 colonnes).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd

    nrows_total = None
    for pfile in parquet_list:
        meta = pq.read_metadata(pfile)
        length = meta.num_rows
        if nrows_total is None:
            nrows_total = length
        else:
            nrows_total = min(nrows_total, length)

    if nrows_total is None:
        print("[WARN] Aucun fichier parquet dans merge_parquets_out_of_core")
        return

    base_dir = os.path.dirname(out_parquet) if os.path.dirname(out_parquet) else "."
    os.makedirs(base_dir, exist_ok=True)

    loaded_dfs = [pd.read_parquet(p) for p in parquet_list]

    writer = None
    start=0
    while start<nrows_total:
        end   = min(start+chunk_size, nrows_total)
        chunk_list=[]
        for df_part in loaded_dfs:
            sub = df_part.iloc[start:end]
            chunk_list.append(sub)
        merged_df = pd.concat(chunk_list, axis=1, ignore_index=False)

        table_merged = pa.Table.from_pandas(merged_df)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table_merged.schema)
        writer.write_table(table_merged)
        start+=chunk_size

    if writer:
        writer.close()
    print(f"[OK] Merge => {out_parquet}")

def load_final_rolling_features(df_keno, windows=[20,50,100],
                                chunk_size=2000, out_prefix="rolling_feats"):
    """
    1) compute_rolling_one_window pour w in windows
    2) merge => rolling_feats_all.parquet
    """
    parquet_paths=[]
    for w in windows:
        out_file = f"{out_prefix}_{w}.parquet"
        compute_rolling_one_window(df_keno, w, chunk_size=chunk_size, out_parquet=out_file)
        parquet_paths.append(out_file)

    out_final = f"{out_prefix}_all.parquet"
    merge_parquets_out_of_core(parquet_paths, out_parquet=out_final, chunk_size=chunk_size)
    return out_final
