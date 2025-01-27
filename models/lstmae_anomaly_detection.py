import torch

from torch import nn

class LSTMAEAnomalyDetector(nn.Module):
    def __init__(self, w_size=10, input_size=8, hidden_size=10):
        super(LSTMAEAnomalyDetector, self).__init__()
        self.w_size = w_size

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size=input_size, hidden_size=(hidden_size*2), batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=(hidden_size*2), hidden_size=hidden_size, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=(hidden_size*2), batch_first=True)
        self.decoder_linear = nn.Linear((hidden_size*2), input_size)

    def forward(self, x):
        # (batch, 1, window, n_bytes)
        x = x.squeeze(1)
        
        # Encoder
        x, (hn, cn) = self.encoder_lstm1(x)
        x, (hn, cn) = self.encoder_lstm2(x)
        x = hn  # Use only the last hidden state for the embedding vector

        # Repeat the embedding vector w times
        x = x.repeat(self.w_size, 1, 1).permute(1, 0, 2)

        # Decoder
        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)
        x = self.decoder_linear(x)

        # (batch, window, n_bytes)
        x = x.unsqueeze(1)

        return x
