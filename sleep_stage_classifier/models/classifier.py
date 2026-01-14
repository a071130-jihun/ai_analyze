from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class SleepStageCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_channels, 32, dropout=dropout * 0.5),
            ConvBlock(32, 64, dropout=dropout * 0.7),
            ConvBlock(64, 128, dropout=dropout),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class SleepStageCRNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_lstm_layers: int = 2
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_channels, 32, dropout=dropout * 0.5),
            ConvBlock(32, 64, dropout=dropout * 0.7),
            ConvBlock(64, 128, dropout=dropout),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, None))
        
        self.lstm = nn.LSTM(
            input_size=128 * 8,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        
        x = x.permute(0, 3, 1, 2)
        time_steps = x.size(1)
        x = x.contiguous().view(batch_size, time_steps, -1)
        
        lstm_out, _ = self.lstm(x)
        
        x = lstm_out[:, -1, :]
        
        x = self.classifier(x)
        return x


class SleepStageTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_channels, 32, dropout=dropout * 0.5),
            ConvBlock(32, 64, dropout=dropout),
        )
        
        self.feature_dim = 64 * 32
        self.proj = nn.Linear(self.feature_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.conv_blocks(x)
        
        x = x.permute(0, 3, 1, 2)
        time_steps = x.size(1)
        x = x.contiguous().view(batch_size, time_steps, -1)
        
        x = self.proj(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        
        cls_output = x[:, 0, :]
        
        x = self.classifier(cls_output)
        return x


class DeepSleepTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_heads: int = 8,
        num_transformer_layers: int = 4
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_channels, 32, dropout=dropout * 0.3),
            ConvBlock(32, 64, dropout=dropout * 0.5),
            ConvBlock(64, 128, dropout=dropout * 0.7),
            ConvBlock(128, 256, dropout=dropout),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        self.feature_dim = 256 * 8
        self.proj = nn.Linear(self.feature_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 65, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.transformer_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)
        
        x = x.permute(0, 3, 1, 2)
        time_steps = x.size(1)
        x = x.contiguous().view(batch_size, time_steps, -1)
        
        x = self.proj(x)
        x = self.proj_norm(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        x = self.transformer(x)
        x = self.transformer_norm(x)
        
        cls_output = x[:, 0, :]
        
        x = self.classifier(cls_output)
        return x


def get_model(model_type: str = "cnn", **kwargs) -> nn.Module:
    from .sequence_model import SleepSequenceModel, DeepSleepResNet, DeepSleepResNetLarge
    
    models = {
        "cnn": SleepStageCNN,
        "crnn": SleepStageCRNN,
        "transformer": SleepStageTransformer,
        "deep_transformer": DeepSleepTransformer,
        "sequence": SleepSequenceModel,
        "deep_resnet": DeepSleepResNet,
        "deep_resnet_large": DeepSleepResNetLarge,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)
