import torch
import torch.nn as nn
from torchsummary import summary

class CRNN(nn.Module):
    def __init__(self, num_classes, drop_out_rate = 0.35):
        super().__init__()
        #CNN
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', bias=True),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=(2,2))
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=True),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=(2,2))
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=True),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', bias=True),
        nn.Dropout(drop_out_rate),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', bias=True),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', bias=True),
        nn.Dropout(drop_out_rate),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))     
        )   
        self.conv7 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding='same', bias=True),
        nn.Dropout(0.25),
        nn.BatchNorm2d(num_features=512),
        nn.ReLU()
        )

        self.fc1 = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU())

        #RNN
        self.rnn1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        #FC
        self.fc2 = nn.Linear(512, num_classes)
        #Softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        #CNN to RNN
        x = x.permute(0, 2, 1, 3)  # swap height and width dimensions
        x = x.reshape(x.shape[0], 16, -1)  # reshape (batch_size, seq_length, num_channels * height) # 16 = max_label_len
        x = self.fc1(x)
       
        x = self.rnn1(x)[0]
        x = self.rnn2(x)[0]
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':    
    input_data = torch.rand(8, 1, 64, 128)
    
    model = CRNN(num_classes=188)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)
        break