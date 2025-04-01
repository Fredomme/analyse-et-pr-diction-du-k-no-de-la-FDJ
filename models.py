import torch
import torch.nn as nn

class LSTMWithEmbedding(nn.Module):
    def __init__(self, vocab_size=71, embed_dim=32, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x_embed = self.embedding(x)
        output, _ = self.lstm(x_embed)
        logits = self.fc(output[:, -1, :])
        return logits

class TransformerWithTime(nn.Module):
    def __init__(self, vocab_size=71, embed_dim=32, hidden_dim=256, num_layers=3, nhead=4, time_feat_dim=16, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(7, time_feat_dim),  # Dimension corrigée à 7 ici
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(time_feat_dim, time_feat_dim),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim + time_feat_dim, vocab_size)

    def forward(self, x, time_feats):
        x_embed = self.embedding(x)
        tf = self.time_mlp(time_feats).unsqueeze(1).repeat(1, x_embed.size(1), 1)
        x_trans = self.transformer_encoder(x_embed)
        x_combined = torch.cat([x_trans[:, -1, :], tf[:, -1, :]], dim=-1)
        logits = self.fc(x_combined)
        return logits

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
