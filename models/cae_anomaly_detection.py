import torch

from torch import nn

class CAEAnomalyDetector(nn.Module):
    def __init__(self, w_size, input_size=58, hidden_size=64):
        super(CAEAnomalyDetector, self).__init__()

        self.w_size = w_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, (hidden_size//4), kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d((hidden_size//4), eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d((hidden_size//4), (hidden_size//2), kernel_size=3, stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d((hidden_size//2), eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d((hidden_size//2), hidden_size, kernel_size=3, stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * w_size, hidden_size * w_size//2), # CHANGE TO THIS WHEN RUNNING SAD: nn.Linear(hidden_size * 4, hidden_size * 4//2), # 
            nn.ReLU()
        )

        # Unembedding
        self.enembedding = nn.Sequential(
            nn.Linear(hidden_size * w_size//2, hidden_size * w_size), # CHANGE TO THIS WHEN RUNNING SAD: nn.Linear(hidden_size*4//2, hidden_size*4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(hidden_size, w_size // 8, 8)), # CHANGE TO THIS WHEN RUNNING SAD: nn.Unflatten(dim=1, unflattened_size=(hidden_size, 2, 2))
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_size, out_channels=(hidden_size//2), kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(hidden_size//2), out_channels=(hidden_size//4), kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(hidden_size//4), out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)), # CHANGE TO THIS WHEN RUNNING SAD: nn.ConvTranspose2d(in_channels=(hidden_size//4), out_channels=1, kernel_size=3, stride=2, padding=(1, 2), output_padding=(1, 1)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        x = self.enembedding(x)
        x = self.decoder(x)
        return x
