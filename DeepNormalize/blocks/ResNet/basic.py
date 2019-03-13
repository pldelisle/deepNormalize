from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
