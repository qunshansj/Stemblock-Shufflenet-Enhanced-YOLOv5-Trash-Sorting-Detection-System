python


class StemBlock(nn.Module):
    def __init__(self, c1, c2):
        super(StemBlock, self).__init__()
        self.stem_1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stem_2a = nn.Conv2d(c2, c2//2, kernel_size=1, stride=1)
        self.stem_2b = nn.Conv2d(c2//2, c2, kernel_size=3, stride=2, padding=1)
        self.stem_2c = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = nn.Conv2d(c2*2, c2, kernel_size=1, stride=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        res1 = self.activation(self.stem_1(x))
        res2_a = self.activation(self.stem_2a(res1))
        res2_b = self.activation(self.stem_2b(res2_a))
        res2_c = self.activation(self.stem_2c(res1))
        cat_res = torch.cat((res2_b, res2_c), dim=1)
        out = self.activation(self.stem_3(cat_res))
        return out
