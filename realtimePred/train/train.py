import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

# 데이터셋 재구성 : 그룹화, 시퀀스 정렬, 타겟 레이블블
class GameGRUDataset(Dataset):
    def __init__(self, df, feature_cols, label_col='home_win'):
        self.games = []
        for game_id, group in df.groupby('game_id'):
            group = group.sort_values(by=['inning', 'home_away'])
            X = group[feature_cols].astype(np.float32).values
            y = group[label_col].iloc[0] 
            self.games.append((X, y))

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        X, y = self.games[idx]
        return torch.tensor(X), torch.tensor(y, dtype=torch.float32)

class GRUWinPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 마지막 ht만 가져와서 출력
        return self.sigmoid(self.fc(out))

def train_gru_model(dataset, input_dim, epochs=50, batch_size=8):
    model = GRUWinPredictor(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            y = y.unsqueeze(1)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

df = pd.read_csv("structured_game_logs_all.csv")

# 제거 피처처
drop_cols = ['game_id', 'home_win']

# 사용할 피처 
feature_cols = [
    'inning', 'home_away', 'home_score', 'away_score', 'score_diff',
    'res_2루타','res_3루타','res_기타','res_땅볼아웃','res_뜬공아웃',
    'res_병살','res_볼넷사구','res_삼진','res_실책','res_안타',
    'res_직선타아웃','res_파울','res_홈런','res_희생'
]


dataset = GameGRUDataset(df, feature_cols, label_col='home_win')
model = train_gru_model(dataset, input_dim=len(feature_cols), epochs=50, batch_size=8)

torch.save(model.state_dict(), 'model/gru_model_full.pt')