import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from convlstm import ConvLSTMCell



class conv_lstm(nn.Module):
    def __init__(self, timesteps = 8, hidden_size=256):
        super().__init__()
        self.timesteps = timesteps
       # self.convlstm = ConvLSTMCell(input_dim=1,
        #         hidden_dim=[64, 64, 128],
         #        kernel_size=(3, 3),
          #       num_layers=3,
           #      batch_first=True
            #     bias=True,)
        
        
        
        self.fc = nn.Linear(hidden_size, self.timesteps * 30 * 54)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.timesteps, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)
            )
        )
    def forward(self, x):
       
        x = self.fc(x)                    # [batch, n-timesteps * 30 * 54]
        x = x.view(x.size(0), self.timesteps, 30, 54)  # [batch, n-timesteps, 30, 54]
        x = self.decoder(x)               # [batch, 1, 120, 217]
        return x

                