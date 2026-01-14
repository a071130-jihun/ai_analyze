import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        mid_ch = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        
        multi_scale = torch.cat([b1, b3, b5, b7], dim=1)
        out = self.fusion(multi_scale)
        out = self.dropout(out)
        
        if self.use_residual:
            out = out + x
        
        out = self.pool(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScaleCNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.stage1 = nn.Sequential(
            MultiScaleConvBlock(32, 64, dropout=dropout * 0.5),
            SEBlock(64)
        )
        
        self.stage2 = nn.Sequential(
            MultiScaleConvBlock(64, 128, dropout=dropout * 0.7),
            SEBlock(128)
        )
        
        self.stage3 = nn.Sequential(
            MultiScaleConvBlock(128, 256, dropout=dropout),
            SEBlock(256)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 256
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


class SleepSequenceModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        seq_len: int = 5,
        predict_center_only: bool = True,
        input_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.predict_center_only = predict_center_only
        
        self.backbone = MultiScaleCNNBackbone(in_channels=input_channels, dropout=dropout)
        self.feature_dim = self.backbone.feature_dim
        
        self.temporal_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 4:
            x = self.backbone(x)
            return self.classifier(
                torch.cat([x, x], dim=-1)
            )
        
        batch_size, seq_len, c, h, w = x.shape
        
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.backbone(x)
        x = x.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.temporal_lstm(x)
        
        if self.predict_center_only:
            center_idx = seq_len // 2
            x = lstm_out[:, center_idx, :]
        else:
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            x = torch.sum(lstm_out * attn_weights, dim=1)
        
        x = self.classifier(x)
        return x


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, seq_len=5, transform=None, subject_ids=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.seq_len = seq_len
        self.transform = transform
        self.subject_ids = subject_ids
        
        if self.features.dim() == 3:
            self.features = self.features.unsqueeze(1)
        
        self.half_seq = seq_len // 2
        self._build_valid_indices()
    
    def _build_valid_indices(self):
        self.valid_indices = []
        
        if self.subject_ids is None:
            for i in range(self.half_seq, len(self.labels) - self.half_seq):
                self.valid_indices.append(i)
        else:
            unique_subjects = []
            current_subject = self.subject_ids[0]
            start_idx = 0
            
            for i, sid in enumerate(self.subject_ids):
                if sid != current_subject:
                    unique_subjects.append((current_subject, start_idx, i))
                    current_subject = sid
                    start_idx = i
            unique_subjects.append((current_subject, start_idx, len(self.subject_ids)))
            
            for _, start, end in unique_subjects:
                for i in range(start + self.half_seq, end - self.half_seq):
                    self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        start_idx = center_idx - self.half_seq
        end_idx = center_idx + self.half_seq + 1
        
        seq_features = self.features[start_idx:end_idx].clone()
        center_label = self.labels[center_idx]
        
        if self.transform:
            transformed = []
            for i in range(len(seq_features)):
                transformed.append(self.transform(seq_features[i]))
            seq_features = torch.stack(transformed)
        
        return seq_features, center_label


def create_sequence_loaders(
    train_features, 
    train_labels, 
    val_features,
    val_labels,
    seq_len=5, 
    batch_size=32, 
    train_transform=None,
    num_workers=4
):
    train_dataset = SequenceDataset(
        train_features, train_labels, 
        seq_len=seq_len, transform=train_transform
    )
    val_dataset = SequenceDataset(
        val_features, val_labels, 
        seq_len=seq_len, transform=None
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader
