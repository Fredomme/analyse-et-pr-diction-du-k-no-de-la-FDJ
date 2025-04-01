import torch
import torch.nn as nn
import csv
import os
from torch.utils.data import Dataset, DataLoader
from models import LSTMWithEmbedding, TransformerWithTime, get_device

##############################################################
# Datasets (LSTM / Transformer) comme dans votre code initial
##############################################################
class KenoShiftDatasetLSTM(Dataset):
    def __init__(self, draws_2d):
        self.draws = draws_2d
        self.N = len(draws_2d)

    def __len__(self):
        return self.N - 1

    def __getitem__(self, idx):
        x_seq = self.draws[idx]
        y_seq = self.draws[idx + 1]
        # Création du vecteur cible de taille 71 (index 0 inutilisé)
        y_vec = torch.zeros(71, dtype=torch.float32)
        for b in y_seq:
            if 1 <= b <= 70:
                y_vec[b] = 1.0
        return torch.tensor(x_seq, dtype=torch.long), y_vec


class KenoShiftDatasetTransformer(Dataset):
    def __init__(self, draws_2d, time_feats):
        self.draws = draws_2d
        self.time_feats = time_feats.values if hasattr(time_feats, 'values') else time_feats
        self.N = len(draws_2d)

    def __len__(self):
        return self.N - 1

    def __getitem__(self, idx):
        x_seq = self.draws[idx]
        y_seq = self.draws[idx + 1]
        tf = self.time_feats[idx + 1]
        # Création du vecteur cible de taille 71 (index 0 inutilisé)
        y_vec = torch.zeros(71, dtype=torch.float32)
        for b in y_seq:
            if 1 <= b <= 70:
                y_vec[b] = 1.0
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(tf, dtype=torch.float32), y_vec


##############################################################
# Entraînement LSTM
##############################################################
def train_model_lstm_embed(train_data, val_data=None, epochs=20, batch_size=32,
                           lr=1e-3, log_csv="training_log_lstm.csv", clip=5.0):
    device = get_device()
    model = LSTMWithEmbedding(vocab_size=71, embed_dim=32, hidden_dim=256,
                              num_layers=3, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # Dataset + DataLoader
    train_dataset = KenoShiftDatasetLSTM(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_data is not None:
        val_dataset = KenoShiftDatasetLSTM(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # CSV log
    if os.path.exists(log_csv):
        os.remove(log_csv)
    f = open(log_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        val_loss_val = -1
        if val_loader:
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits_val = model(x_val)
                    loss_val = criterion(logits_val, y_val)
                    running_val += loss_val.item()
            val_loss_val = running_val / len(val_loader)
            scheduler.step(val_loss_val)

        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss_val:.4f}" if val_loss_val>=0 else "NA"])
        print(f"[LSTM Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss_val if val_loss_val>=0 else 'NA'}")

    f.close()
    return model


##############################################################
# Entraînement Transformer
##############################################################
def train_model_transformer(train_data, train_time, val_data=None, val_time=None,
                            epochs=20, batch_size=32, lr=1e-3, log_csv="training_log_transformer.csv", clip=5.0):
    device = get_device()
    model = TransformerWithTime(
        vocab_size=71,
        embed_dim=32,
        nhead=4,
        num_layers=3,
        hidden_dim=256,
        time_feat_dim=16,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = KenoShiftDatasetTransformer(train_data, train_time)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if (val_data is not None) and (val_time is not None):
        val_dataset = KenoShiftDatasetTransformer(val_data, val_time)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # CSV log
    if os.path.exists(log_csv):
        os.remove(log_csv)
    f = open(log_csv, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, tf_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            tf_batch = tf_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch, tf_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        val_loss_val = -1
        if val_loader:
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for x_val, tf_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    tf_val = tf_val.to(device)
                    y_val = y_val.to(device)
                    logits_val = model(x_val, tf_val)
                    loss_val = criterion(logits_val, y_val)
                    running_val += loss_val.item()
            val_loss_val = running_val / len(val_loader)
            scheduler.step(val_loss_val)

        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss_val:.4f}" if val_loss_val>=0 else "NA"])
        print(f"[Transformer Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss_val if val_loss_val>=0 else 'NA'}")

    f.close()
    return model
