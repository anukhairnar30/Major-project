import torch
import torch.nn as nn
import timm


# ============================================================
# BiLSTM + Attention Head
# ============================================================

class BiLSTMAttentionHead(nn.Module):

    def __init__(
        self,
        embed_dim,
        meta_dim,
        lstm_hidden=384,
        num_classes=7
    ):
        super().__init__()

        self.feature_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Sequential(
            nn.Linear(
                lstm_hidden * 2,
                lstm_hidden
            ),
            nn.Tanh(),
            nn.Linear(
                lstm_hidden,
                1
            )
        )

        fused_dim = (
            (lstm_hidden * 2)
            + embed_dim
            + meta_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),

            nn.Dropout(0.4),

            nn.Linear(
                fused_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        patch_tokens,
        cls_token,
        meta_features
    ):

        x = self.feature_dropout(
            patch_tokens
        )

        lstm_out, _ = self.lstm(x)

        attn_scores = self.attn(
            lstm_out
        )

        attn_weights = torch.softmax(
            attn_scores,
            dim=1
        )

        context = (
            lstm_out * attn_weights
        ).sum(
            dim=1
        )

        fused_features = torch.cat(
            (
                context,
                cls_token,
                meta_features
            ),
            dim=1
        )

        return self.classifier(
            fused_features
        )


# ============================================================
# Multimodal Hybrid ViT
# ============================================================

class MultimodalHybridViT(nn.Module):

    def __init__(
        self,
        num_classes=7,
        meta_dim=19,
        lstm_hidden=384
    ):
        super().__init__()

        # Exact backbone used during training
        self.backbone = timm.create_model(
            "vit_base_patch16_384",
            pretrained=True,
            num_classes=0
        )

        embed_dim = self.backbone.num_features

        self.head = BiLSTMAttentionHead(
            embed_dim=embed_dim,
            meta_dim=meta_dim,
            lstm_hidden=lstm_hidden,
            num_classes=num_classes
        )

    def forward(
        self,
        images,
        meta_features
    ):

        tokens = self.backbone.forward_features(
            images
        )

        # CLS token
        cls_token = tokens[:, 0, :]

        # Patch tokens
        patch_tokens = tokens[:, 1:, :]

        return self.head(
            patch_tokens,
            cls_token,
            meta_features
        )


# ============================================================
# Load Trained Model
# ============================================================

def load_trained_model(
    model_path,
    device=None
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = MultimodalHybridViT(
        num_classes=7,
        meta_dim=19,
        lstm_hidden=384
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model = model.to(device)

    model.eval()

    return model