import numpy as np
import torch.nn as nn


class SimpleModel(nn.Module):
    '''
    Only for non-image data, as not containing any convolutional layers
    '''
    def __init__(self, config):
        super(SimpleModel, self).__init__()

        self.in_channels = config.in_channels if hasattr(config, 'in_channels') else 3

    def forward(self):
        pass

