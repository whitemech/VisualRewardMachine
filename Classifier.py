import torch.nn as nn
import torch.nn.functional as F
import torch
sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x/temp)

class CNN(nn.Module):
    
    def __init__(self, channels=3, classes=5, nodes_linear=4704, mutually_exc=True):
        super().__init__()
        self.classes = classes
        if mutually_exc:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid(inplace = False)
        self.conv1 = nn.Conv2d(channels, 3, 7, stride=2)

        self.conv2 = nn.Conv2d(3, 6, 7, stride=2)
    
        self.fc1 = nn.Linear(nodes_linear, classes)

        
    def forward(self, x):
        

        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x =self.fc1(x)

        #softmax
        return self.activation(x)


class CNN_feature_extraction(nn.Module):

    def __init__(self, channels=3, output_dim=5, nodes_linear=4704):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
             nn.Conv2d(channels, 3, 7, stride=2),
             nn.ReLU(inplace = False),
             nn.Conv2d(3, 6, kernel_size=7, stride=2),
             nn.ReLU(inplace = False),
             nn.Flatten()
        )
        self.fully_connected = nn.Sequential(
             nn.Linear(nodes_linear, output_dim)
        )

    def forward(self, x):
         x = self.cnn_encoder(x)
         return self.fully_connected(x)

class Linear_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = F.softmax(self.lin1(x), dim=-1)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x