import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAttention(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(context_dim, hidden_dim)
        self.value = nn.Linear(context_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, h, ctx_feats):
        # h: [B, H], ctx_feats: [B, C]
        q = self.query(h).unsqueeze(1)           # [B,1,H]
        k = self.key(ctx_feats).unsqueeze(1)     # [B,1,H]
        v = self.value(ctx_feats).unsqueeze(1)   # [B,1,H]
        scores = (q * k).sum(-1) / self.scale    # [B,1]
        w = F.softmax(scores, dim=1).unsqueeze(-1)  # [B,1,1]
        attended = (w * v).sum(1)               # [B,H]
        return attended

class ContextualSocialLSTM(nn.Module):
    def __init__(
        self,
        input_size=9,
        context_dim=15,
        hidden_size=256,
        output_size=2,
        intention_dim=1,
        num_layers=2
    ):
        super().__init__()
        self.ego_lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers, batch_first=True
        )
        self.nbr_lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers, batch_first=True
        )
        self.context_attention = ContextAttention(context_dim, hidden_size)
        self.intention_fc = nn.Linear(intention_dim, hidden_size)
        # we’ll concat [ego, nbr, ctx, intent] each H-dimensional → 4H
        self.output_fc    = nn.Linear(hidden_size * 4, output_size)

    def forward(self, hist, nbrs, mask, context, intention):
        """
        hist: [B, seq_len, 9]
        nbrs: [B, N, 1, 9]   (after expand_dims)
        mask: [B, N, 1]
        context: [B, context_dim]
        intention: [B,1]
        """
        B, N, T, Ff = nbrs.size()
        # flatten neighbors into batch
        nbr_flat, m_flat = (
            nbrs.view(B*N, T, Ff),
            mask.view(B*N, 1)
        )

        nbr_out, _ = self.nbr_lstm(nbr_flat)             # [B*N, T, H]
        nbr_final  = nbr_out[:, -1, :].view(B, N, -1)    # [B, N, H]

        # zero-out masked neighbors
        nbr_final = nbr_final * mask                     # [B, N, H]
        nbr_pooled = nbr_final.sum(1) / (mask.sum(1) + 1e-6)  # [B, H]

        # ego LSTM
        ego_out, _ = self.ego_lstm(hist)                 # [B, seq_len, H]
        ego_final  = ego_out[:, -1, :]                   # [B, H]

        # context attention
        ctx_att    = self.context_attention(ego_final, context)  # [B, H]

        # intention embedding
        int_emb    = F.relu(self.intention_fc(intention))       # [B, H]

        # fuse and project
        fused = torch.cat([ego_final, nbr_pooled, ctx_att, int_emb], dim=1)  # [B,4H]
        return self.output_fc(fused)

def apply_weight_init(model):
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(_init)
