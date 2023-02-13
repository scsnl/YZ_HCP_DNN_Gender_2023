import torch.nn as nn

# Create the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #CNN Block 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 256, kernel_size=7, stride=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # CNN Block 1
        self.layer4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, stride=1),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=2))
        # Dropout
        self.drop_out = nn.Dropout(p=0.5)
        # Output FC Layer
        self.fc1 = nn.Linear(512, 2)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.layer5(out)
        out = self.layer6(out)
        # Temporal Averaging
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = self.fc1(out)
        return out