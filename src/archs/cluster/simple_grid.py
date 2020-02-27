import numpy as np
import torch.nn as nn
import torch


class SimpleModel(nn.Module):
    '''
    Only for non-image 2D-data, as not containing any convolutional layers
    '''
    def __init__(self, config, num_classes):
        super(SimpleModel, self).__init__()

        self.fc = torch.nn.Linear(2, num_classes)

    def forward(self, x):
        print('x.size()', x.size())
        assert(len(x.size()) == 3)

        x = self.fc(x)
        return x



