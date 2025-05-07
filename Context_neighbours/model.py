
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

    def forward(self, decoder_hidden, context_features):
        # decoder_hidden: [B,H], context_features: [B,ctx_dim]
        q = self.query(decoder_hidden).unsqueeze(1)       # [B,1,H]
        k = self.key(context_features).unsqueeze(1)       # [B,1,H]
        v = self.value(context_features).unsqueeze(1)     # [B,1,H]
        scores = (q * k).sum(-1) / self.scale             # [B,1]
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B,1,1]
        attended = (weights * v).squeeze(1)               # [B,H]
        return attended

class ContextualSocialLSTM(nn.Module):
    def __init__(
        self,
        input_size=9,
        context_dim=15,
        hidden_size=256,
        output_size=2,
        intention_dim=1,
        num_layers=2,
        dist_temp=1.0
    ):
        super().__init__()
        self.ego_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.nbr_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.context_attention = ContextAttention(context_dim, hidden_size)
        self.intention_fc      = nn.Linear(intention_dim, hidden_size)
        # we will concat [ego, nbr_agg, context_attn, intention_emb] => 4 * hidden_size
        self.output_fc = nn.Linear(hidden_size * 4, output_size)

        self.dist_temp = dist_temp

    def forward(self, hist, nbrs, mask, context, intention, nbr_pos, ego_pos):
        """
        nbrs: [B,N,1,feat_dim]  (we assume time-dim=1)
        mask: [B,N,1]
        nbr_pos: [B,N,2]
        ego_pos: [B,2]
        """
        B, N, T, Fdim = nbrs.size()
        # neighbor LSTM over the 1-step 'sequence'
        nbrs_flat = nbrs.view(B * N, T, Fdim)
        nbr_out, _ = self.nbr_lstm(nbrs_flat)            # [B*N, T, H]
        nbr_final  = nbr_out[:, -1, :].view(B, N, -1)    # [B,N,H]

        # distance-based weights
        # compute Euclid distance in (x,y)
        dists = torch.norm(nbr_pos - ego_pos.unsqueeze(1), dim=-1)  # [B,N]
        mask_n = mask.squeeze(-1)                                    # [B,N]
        # push invalid neighbors far away
        large = 1e6
        dists = dists + (1.0 - mask_n) * large
        # negative distance weighting
        w = torch.exp(-dists / self.dist_temp) * mask_n             # [B,N]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)                 # normalize
        nbr_agg = torch.sum(nbr_final * w.unsqueeze(-1), dim=1)     # [B,H]

        # ego LSTM
        ego_out, _ = self.ego_lstm(hist)                            # [B,T,H]
        ego_final  = ego_out[:, -1, :]                              # [B,H]

        # contextual attention
        ctx_attn = self.context_attention(ego_final, context)      # [B,H]

        # intention embedding
        intent_emb = F.relu(self.intention_fc(intention.unsqueeze(1))).squeeze(1)  # [B,H]

        # concatenate and predict
        combined = torch.cat([ego_final, nbr_agg, ctx_attn, intent_emb], dim=1)  # [B,4H]
        out = self.output_fc(combined)                                           # [B,2]
        return out

def apply_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
