import torch
import torch.nn as nn

class SocialLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=256, output_size=2, num_layers=2):
        super(SocialLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output size depends on the task (e.g., 2 for x, y)

    def forward(self, hist, nbrs, mask):
        # Check if hist and nbrs have the same sequence length
        hist_size = hist.size()
        nbrs_size = nbrs.size()


        if hist_size[1] != nbrs_size[1]:
            # If sequence lengths don't match, we pad the smaller tensor
            if hist_size[1] > nbrs_size[1]:
                # Pad nbrs along the sequence dimension
                padding_size = hist_size[1] - nbrs_size[1]
                nbrs = torch.cat([nbrs, torch.zeros(nbrs_size[0], padding_size, nbrs_size[2])], dim=1)
            elif nbrs_size[1] > hist_size[1]:
                # Pad hist along the sequence dimension
                padding_size = nbrs_size[1] - hist_size[1]
                hist = torch.cat([hist, torch.zeros(hist_size[0], padding_size, hist_size[2])], dim=1)

        # Now hist and nbrs should have the same sequence length
        x = torch.cat([hist, nbrs], dim=2)  # Concatenate along the feature dimension

        # Pass through LSTM
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last hidden state for prediction
        return x

# Weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Function to apply weight initialization to the model
def apply_weight_init(model):
    model.apply(init_weights)
