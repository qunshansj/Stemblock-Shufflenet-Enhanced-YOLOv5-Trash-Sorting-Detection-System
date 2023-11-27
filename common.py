python


class CRM(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super(CRM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        res = self.mp(self.conv1(x))
        return res

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = x.permute(0, 2, 1, 3, 4).contiguous()

    x = x.view(batch_size, -1, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = inp // 2
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if self.stride == 2:
            layers.append(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False))
            layers.append(nn.BatchNorm2d(inp))
            layers.append(nn.ReLU(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
