import torch
import torch.nn as nn
from torchsummaryX import summary

# class Telnet(nn.Module):
#
#     def __init__(self):
#         super(Telnet, self).__init__()
#         self.linear = torch.nn.Linear(4, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         y_pred = self.sigmoid(self.linear(x))
#         return y_pred

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.hl_1 = nn.Sequential(nn.Linear(in_features=196, out_features=98, bias=True),
                                  nn.Dropout(0.1),
                                  nn.ReLU())

        self.hl_2 = nn.Sequential(nn.Linear(in_features=98, out_features=49, bias=True),
                                  nn.Dropout(0.1),
                                  nn.ReLU())

        self.hl_3 = nn.Sequential(nn.Linear(in_features=49, out_features=10, bias=True),
                                  nn.Dropout(0.3),
                                  nn.ReLU())

        self.cl = nn.Sequential(nn.Linear(in_features=10, out_features=1, bias=True))

    def forward(self, x):
        fc1 = self.hl_1(x)
        fc2 = self.hl_2(fc1)
        fc3 = self.hl_3(fc2)
        output = self.cl(fc3)
        return output

if __name__ == '__main__':
    net = MLPClassifier()
    net(torch.ones(1,1,196))
    summary(net,torch.ones(1,1,196))