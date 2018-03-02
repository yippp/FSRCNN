import torch
import torch.nn as nn
from math import sqrt


class Net(torch.nn.Module):
    def __init__(self, n_channels, d=56, s=12, m=4):
        # too big network may leads to over-fitting
        super(Net, self).__init__()

        # Feature extraction
        self.first_part = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=d, kernel_size=3, stride=1, padding=0),
                                        nn.PReLU())
        # H_out = floor((H_in+2*padding-(kernal_size-1)-1)/stride+1)
        #       = floor(H_in-4)
        # for x2  floor(H_in-2)
        self.layers = []
        # Shrinking
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        # Non-linear Mapping
        for _ in range(m):
            self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1),
                                         nn.PReLU()))

        # # Expanding
        self.layers.append(nn.Sequential(nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, stride=1, padding=0),
                                         nn.PReLU()))

        self.mid_part = torch.nn.Sequential(*self.layers)

        # Deconvolution
        # self.last_part = nn.ConvTranspose2d(in_channels=d, out_channels=n_channels, kernel_size=9, stride=3, padding=4, output_padding=0)
        self.last_part = nn.Sequential(nn.Conv2d(in_channels=d, out_channels=n_channels * 2 * 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelShuffle(2))
        # H_out = (H_in-1)*stride-2*padding+kernal_size+out_padding
        #       = (H_in-1)*3+1
        #test input should be (y-5)*3+1
        # for x2 2x-3
        # for x4 4x-25

    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0.0, 0.2)
                m.weight.data.normal_(0.0, sqrt(2/m.out_channels/m.kernel_size[0]/m.kernel_size[0])) # MSRA
                # nn.init.xavier_normal(m.weight) # Xavier
                if m.bias is not None:
                    m.bias.data.zero_()