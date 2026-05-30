import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch



class radar_cnn(nn.Module):
    def __init__(self, timesteps = 8, hidden_size=256):
        super().__init__()
        self.timesteps = timesteps
        
        
        
        self.conv1 = nn.Conv2d(self.timesteps,12,5,padding=5//2)
        self.gap = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(12,24,5,padding=5//2)
        self.conv3 = nn.Conv2d(24,24,5,padding=5//2)
       
        self.conv4 = nn.Conv2d(24,36,5,padding=5//2)
        self.conv5 = nn.Conv2d(36,48,5,padding=5//2)
        
        self.lstm = nn.LSTM(
            input_size=91,      # may need changing
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=10
        )
        
        self.fc = nn.Linear(hidden_size, self.timesteps * 30 * 54)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.timesteps, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 1, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)
            ),
            #nn.Sigmoid()
        )
    def forward(self, x):
        x= self.conv1(x)
        x= self.gap(x)
        x=self.conv2(x)
        x=self.gap(x)
        x=self.conv3(x)
        
        x=self.gap(x)
        x=self.conv4(x)
        x=self.gap(x)
        x=self.conv5(x)
        x= torch.flatten(x,2)
       
        lstm_out, (h_n, c_n)= self.lstm(x)
        x = lstm_out[:, -1, :]  
        x = self.fc(x)                    # [batch, n-timesteps * 30 * 54]
        x = x.view(x.size(0), self.timesteps, 30, 54)  # [batch, n-timesteps, 30, 54]
        x = self.decoder(x)               # [batch, 1, 120, 217]
        return x

                