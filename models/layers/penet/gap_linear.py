import torch.nn as nn
import torch

class GAPLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Global average pooling (3D) followed by a linear layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels
        """
        super(GAPLinear, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # self.prompt_ft = nn.Linear(in_channels+63,in_channels+63)
        self.fc = nn.Linear(in_channels, out_channels)
        self.fc.is_output_head = True

    def forward(self, x, prompt=None):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if prompt != None:
            x = torch.concat((prompt,x),dim=1)
            x = self.prompt_ft(x)[:,63:]
            x = self.fc(x)
        else:
            x = self.fc(x)

        return x
