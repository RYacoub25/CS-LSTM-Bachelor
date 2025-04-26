# model.py
import torch
import torch.nn as nn

class ContextAttention(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super(ContextAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(context_dim, hidden_dim)
        self.value = nn.Linear(context_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, decoder_hidden, context_features):
        """
        decoder_hidden: [batch_size, hidden_dim]
        context_features: [batch_size, context_dim]
        """
        q = self.query(decoder_hidden).unsqueeze(1)  # [B, 1, H]
        k = self.key(context_features).unsqueeze(1)  # [B, 1, H]
        v = self.value(context_features).unsqueeze(1)  # [B, 1, H]

        attn_scores = (q * k).sum(dim=-1) / self.scale  # [B, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, 1]

        attended_context = attn_weights.unsqueeze(-1) * v  # [B, 1, H]
        attended_context = attended_context.squeeze(1)  # [B, H]
        return attended_context

class ContextualSocialLSTM(nn.Module):
    def __init__(self, input_size=9, context_dim=15, hidden_size=256, output_size=2, num_layers=2):
        super(ContextualSocialLSTM, self).__init__()
        self.ego_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        self.nbr_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
        
        self.context_attention = ContextAttention(context_dim, hidden_size)
        self.output_fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, hist, nbrs, mask, context):
        if nbrs.dim() == 3:
            # Already (batch_size, seq_len, feat_dim)
            nbrs = nbrs.unsqueeze(1)  # -> (batch_size, 1, seq_len, feat_dim)

        batch_size, num_neighbors, seq_len, feat_dim = nbrs.size()
        nbrs = nbrs.view(batch_size * num_neighbors, seq_len, feat_dim)

        nbr_out, _ = self.nbr_lstm(nbrs)
        nbr_final = nbr_out[:, -1, :].view(batch_size, num_neighbors, -1)

        attended_nbrs = nbr_final.mean(dim=1)  # Simplified mean pooling instead of neighbor attention

        ego_out, _ = self.ego_lstm(hist)
        ego_final = ego_out[:, -1, :]

        attended_context = self.context_attention(ego_final, context)

        combined = torch.cat([ego_final + attended_nbrs, attended_context], dim=1)
        output = self.output_fc(combined)
        return output

# Weight Initialization (keep as you had)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def apply_weight_init(model):
    model.apply(init_weights)
