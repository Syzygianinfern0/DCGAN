from config import *
from weight_init import *


class Generator(nn.Module):
    def __init__(self):
        """
        Init nn.Sequential object for Generator Network
        """
        super(Generator, self).__init__()

        strides = [1, 2, 2, 2, 2]
        paddings = [0, 1, 1, 1, 1]
        channels = [nz,
                    ngf * 8,
                    ngf * 4,
                    ngf * 2,
                    ngf,
                    nc]

        blocks = [self.trans_conv_block(in_c, out_c, stride, padding)
                  for in_c, out_c, stride, padding in
                  zip(channels[:-2], channels[1:], strides, paddings)]
        blocks.append(self.trans_conv_block(channels[-2], channels[-1], strides[-1], paddings[-1], act='tanh'))

        self.generator = nn.Sequential(*list(blocks))

    def forward(self, x):
        return self.generator(x)

    @staticmethod
    def trans_conv_block(in_c, out_c, stride, padding, act='relu', *args, **kwargs):
        if act.lower() == 'relu':
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                   stride=stride, padding=padding,
                                   bias=False, *args, **kwargs),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
        elif act.lower() == 'tanh':
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                   stride=stride, padding=padding,
                                   bias=False, *args, **kwargs),
                nn.Tanh()
            )


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        strides = [2, 2, 2, 2, 1]
        paddings = [1, 1, 1, 1, 0]
        channels = [nc,
                    ndf,
                    ndf * 2,
                    ndf * 4,
                    ndf * 8,
                    1]
        bns = [False, True, True, True, True]

        blocks = [self.trans_conv_block(in_c, out_c, stride, padding, bn=bn)
                  for in_c, out_c, stride, padding, bn in
                  zip(channels[:-2], channels[1:], strides, paddings, bns)]

        blocks.append(self.trans_conv_block(channels[-2], channels[-1], strides[-1],
                                            paddings[-1], act='sigmoid', bn=bns[-1]))

        self.discriminator = nn.Sequential(*list(blocks))

    def forward(self, x):
        return self.discriminator(x)

    @staticmethod
    def trans_conv_block(in_c, out_c, stride, padding, act='relu', bn=True, *args, **kwargs):
        if act.lower() == 'relu':
            if bn is True:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=4,
                              stride=stride, padding=padding,
                              bias=False, *args, **kwargs),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(0.2)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=4,
                              stride=stride, padding=padding,
                              bias=False, *args, **kwargs),
                    nn.LeakyReLU(0.2)
                )
        elif act.lower() == 'sigmoid':
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4,
                                   stride=stride, padding=padding,
                                   bias=False, *args, **kwargs),
                nn.Sigmoid()
            )


if __name__ == '__main__':
    netG: nn.Module = Generator()
    netG.to(device)
    if device.type == 'cuda':
        netG = nn.DataParallel(netG, [0])
    netG.apply(weights_init)
    print(netG)

    netD: nn.Module = Discriminator()
    netD.to(device)
    if device.type == 'cuda':
        netG = nn.DataParallel(netD, [0])
    netD.apply(weights_init)
    print(netD)
