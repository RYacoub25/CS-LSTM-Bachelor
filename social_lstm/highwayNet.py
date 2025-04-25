# ==== highwayNet.py ====
import torch
import torch.nn as nn
import torch.nn.functional as F

class highwayNet(nn.Module):
    def __init__(self, args):
        super(highwayNet, self).__init__()

        self.use_cuda = args['use_cuda']
        self.train_flag = args['train_flag']

        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']
        self.input_embedding_size = args['input_embedding_size']

        self.ip_emb = nn.Linear(2, self.input_embedding_size)
        self.enc_lstm1 = nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        self.enc_lstm2 = nn.LSTM(self.input_embedding_size, self.encoder_size, 1)
        self.spatial_embedding = nn.Linear(self.encoder_size, self.encoder_size)

        self.pre4att = nn.Sequential(
            nn.Linear(self.encoder_size, 1)
        )

        self.dec_lstm = nn.LSTM(self.encoder_size, self.decoder_size)
        self.op = nn.Linear(self.decoder_size, 5)

    def attention(self, weights, lstm_out):
        alpha = F.softmax(weights, dim=1)
        lstm_out = lstm_out.permute(0, 2, 1)
        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)
        return F.relu(new_hidden_state), alpha

    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):
        lstm_out, (hist_enc, _) = self.enc_lstm1(F.leaky_relu(self.ip_emb(hist)))
        hist_enc = hist_enc.squeeze().unsqueeze(2)

        nbrs_out, (nbrs_enc, _) = self.enc_lstm2(F.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_enc.squeeze(0).view(nbrs_enc.shape[1], nbrs_enc.shape[2])

        soc_enc = torch.zeros_like(masks).float()
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)

        soc_enc = soc_enc.permute(0, 3, 2, 1)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)

        new_hs = torch.cat((soc_enc, hist_enc), 2)
        new_hs_per = new_hs.permute(0, 2, 1)

        weight = self.pre4att(torch.tanh(new_hs_per))
        new_hidden_ha, _ = self.attention(weight, new_hs_per)

        enc = new_hidden_ha.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)

        return fut_pred, None
