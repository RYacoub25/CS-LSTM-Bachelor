import torch
import torch.nn as nn

class highwayNet(nn.Module):
    def __init__(self, args):
        super(highwayNet, self).__init__()
        self.encoder = nn.LSTM(input_size=9, hidden_size=args['encoder_size'], batch_first=True)
        self.decoder = nn.Linear(args['encoder_size'], 2)  # Predict single (x, y)

    def forward(self, hist, nbrs=None, masks=None, lat_enc=None, lon_enc=None):
        _, (h, _) = self.encoder(hist)
        h = h.squeeze(0)
        out = self.decoder(h)
        return out, None
