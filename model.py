import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

class CnnFeatureExtractor(nn.Module):
    def __init__(self, input_channels=2, cir_len=128, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feature_dim)
        self.relu_fc = nn.ReLU()

        logging.info(f"CNN Feature Extractor initialized. Output feature dimension: {feature_dim}")

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu_fc(self.fc(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.float32, device=x.device).unsqueeze(1)
        i = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device)
        div_term = torch.exp(i * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(x.size(1), self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return x + pe.unsqueeze(0)

class MultiHeadAttentionAggregation(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.tdoa_scale = nn.Parameter(torch.tensor(0.5))
        logging.info("Multi-Head Attention Aggregation module initialized.")

    def forward(self, features, mask, tdoa=None):
        if tdoa is not None:
            tdoa_scaled = tdoa.unsqueeze(-1)
            features = features - self.tdoa_scale * tdoa_scaled

        key_padding_mask = mask == 0
        attn_output, attn_weights = self.multihead_attn(features, features, features, key_padding_mask=key_padding_mask)
        aggregated_features = attn_output.sum(dim=1)
        return aggregated_features, attn_weights

class MeanAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, mask, tdoa=None):
        mask = mask.unsqueeze(-1)
        masked_features = features * mask
        sum_features = masked_features.sum(dim=1)
        valid_counts = mask.sum(dim=1).clamp(min=1e-6)
        aggregated_features = sum_features / valid_counts
        attn_weights = None
        return aggregated_features, attn_weights

class AnchorCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query_feat, anchor_emb, anchor_mask):
        gated_emb = anchor_emb * self.gate(anchor_emb)
        key_padding_mask = anchor_mask == 0
        attn_output, _ = self.cross_attn(query_feat, gated_emb, gated_emb, key_padding_mask=key_padding_mask)
        return attn_output

class LocalizationModel(nn.Module):
    def __init__(self, cir_len=128, num_anchors=8, anchor_pos_dim=2,
                 cnn_feature_dim=128, use_anchor_pos=True, pos_embed_dim=16,
                 use_tdoa=True, tdoa_embed_dim=16, agg_attention_dim=64,
                 mlp_hidden_dim=256, dropout_rate=0.3, num_heads=4, use_attention=True):
        super().__init__()
        self.num_anchors = num_anchors
        self.use_anchor_pos = use_anchor_pos
        self.use_tdoa = use_tdoa

        self.cnn_extractor = CnnFeatureExtractor(input_channels=2, cir_len=cir_len, feature_dim=cnn_feature_dim)
        current_feature_dim = cnn_feature_dim

        if self.use_anchor_pos:
            self.anchor_pos_embedder = nn.Sequential(
                nn.Linear(anchor_pos_dim, 64),
                nn.ReLU(),
                nn.Linear(64, pos_embed_dim),
                nn.ReLU()
            )
            self.cross_attn = AnchorCrossAttention(pos_embed_dim, num_heads=num_heads)
            self.cir_proj_for_crossattn = nn.Linear(cnn_feature_dim, pos_embed_dim)
            self.cross_attn_out_proj = nn.Linear(pos_embed_dim, cnn_feature_dim)
            current_feature_dim += pos_embed_dim
        else:
            self.anchor_pos_embedder = None
            self.cross_attn = None

        if self.use_tdoa:
            self.tdoa_embedder = nn.Sequential(
                nn.Linear(1, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, tdoa_embed_dim), nn.ReLU()
            )
            self.tdoa_pos_encoder = PositionalEncoding(tdoa_embed_dim)
            current_feature_dim += tdoa_embed_dim
        else:
            self.tdoa_embedder = None

        if use_attention:
            self.aggregator = MultiHeadAttentionAggregation(feature_dim=current_feature_dim, num_heads=num_heads)
        else:
            self.aggregator = MeanAggregation()

        self.regression_head = nn.Sequential(
            nn.Linear(current_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, 2)
        )

    def forward(self, batch):
        cir_data = batch['cir_data']
        anchor_positions = batch['anchor_positions']
        anchor_mask = batch['anchor_mask']

        if self.use_tdoa:
            if 'tdoa' not in batch:
                raise ValueError("TDoA input is required but not provided in the batch.")
            tdoa = batch['tdoa']
            tdoa = tdoa - tdoa.mean(dim=1, keepdim=True)
            tdoa = tdoa / (tdoa.std(dim=1, keepdim=True) + 1e-8)
        else:
            tdoa = None

        B, N_a, C_in, L = cir_data.shape
        cir_data_reshaped = cir_data.view(B * N_a, C_in, L)
        cir_features = self.cnn_extractor(cir_data_reshaped).view(B, N_a, -1)

        combined_features = [cir_features]

        if self.use_anchor_pos and self.anchor_pos_embedder is not None:
            if anchor_positions.dim() == 2 and anchor_positions.shape[0] == N_a:
                anchor_positions = anchor_positions.unsqueeze(0).expand(B, -1, -1)
            anchor_pos_emb = self.anchor_pos_embedder(anchor_positions.view(B * N_a, -1)).view(B, N_a, -1)

            cir_proj = self.cir_proj_for_crossattn(cir_features)
            cross_attn_out = self.cross_attn(cir_proj, anchor_pos_emb, anchor_mask)
            cross_attn_out = self.cross_attn_out_proj(cross_attn_out)
            cir_features = cir_features + cross_attn_out  # Residual connection
            combined_features = [cir_features, anchor_pos_emb]

        if self.use_tdoa and self.tdoa_embedder is not None and tdoa is not None:
            tdoa_emb = self.tdoa_embedder(tdoa.unsqueeze(-1).view(B * N_a, -1)).view(B, N_a, -1)
            tdoa_emb = self.tdoa_pos_encoder(tdoa_emb)
            combined_features.append(tdoa_emb)

        full_feature = torch.cat(combined_features, dim=2)
        aggregated_features, attn_weights = self.aggregator(full_feature, anchor_mask, tdoa if self.use_tdoa else None)

        final_input = aggregated_features
        predicted_position = self.regression_head(final_input)
        return predicted_position, attn_weights
