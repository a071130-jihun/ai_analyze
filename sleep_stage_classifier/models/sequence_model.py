import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class SleepSequenceModel(nn.Module):
    """
    논문 기반: 여러 연속 에포크를 입력받아 시간적 컨텍스트 활용
    Input: (batch, seq_len, 1, n_mels, time)
    Output: (batch, seq_len, num_classes) or (batch, num_classes) for center
    """
    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        seq_len: int = 5,
        predict_center_only: bool = True
    ):
        super().__init__()
        self.seq_len = seq_len
        self.predict_center_only = predict_center_only
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.feature_dim = 128 * 4 * 4
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(x)
        
        if self.predict_center_only:
            center_idx = seq_len // 2
            x = lstm_out[:, center_idx, :]
            x = self.classifier(x)
        else:
            x = self.classifier(lstm_out)
        
        return x


class SequenceDataset(torch.utils.data.Dataset):
    """연속 에포크를 시퀀스로 묶는 데이터셋"""
    def __init__(self, features, labels, seq_len=5, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
        self.transform = transform
        
        if self.features.dim() == 3:
            self.features = self.features.unsqueeze(1)
        
        self.half_seq = seq_len // 2
        self.valid_indices = list(range(self.half_seq, len(labels) - self.half_seq))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        start_idx = center_idx - self.half_seq
        end_idx = center_idx + self.half_seq + 1
        
        seq_features = self.features[start_idx:end_idx]
        center_label = self.labels[center_idx]
        
        if self.transform:
            transformed = []
            for i in range(len(seq_features)):
                transformed.append(self.transform(seq_features[i]))
            seq_features = torch.stack(transformed)
        
        return seq_features, center_label


def create_sequence_loaders(features, labels, seq_len=5, batch_size=32, val_ratio=0.15, use_augmentation=True):
    """시퀀스 데이터 로더 생성"""
    from torch.utils.data import DataLoader
    import numpy as np
    
    try:
        from sleep_stage_classifier.augmentation import get_train_transform
        train_transform = get_train_transform() if use_augmentation else None
    except:
        train_transform = None
    
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_ratio)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_dataset = SequenceDataset(
        features[train_idx], labels[train_idx], 
        seq_len=seq_len, transform=train_transform
    )
    val_dataset = SequenceDataset(
        features[val_idx], labels[val_idx], 
        seq_len=seq_len, transform=None
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader
