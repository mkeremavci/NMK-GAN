from torch import nn
from collections import OrderedDict


class AutoEncoder(nn.Module):
    def __init__(self,
            in_channels = 1,
            out_channels = 1,
            features = 128,
            in_shape = (64, 64),
            ):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(in_channels, features, in_shape)
        self.decoder = Decoder(out_channels, features, in_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
            in_channels = 1,
            features = 128,
            in_shape = (64, 64),
            ):
        super(Encoder, self).__init__()

        self.in_shape = in_shape
        self.features = features

        self.encoder1 = Encoder._block(in_channels, features//4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Encoder._block(features//4, features//2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Encoder._block(features//2, features)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(
            in_features = (in_shape[0] // 8) * (in_shape[1] // 8) * features,
            out_features = features, 
            )

    def forward(self, x):
        x = self.pool1(self.encoder1(x))
        x = self.pool2(self.encoder2(x))
        x = self.pool3(self.encoder3(x))
        x = x.view(-1, (self.in_shape[0] // 8) * (self.in_shape[1] // 8) * self.features)

        return self.fc(x)

    @staticmethod
    def _block(
            in_channels,
            features,
            kernel_size = 3,
            ):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv1',
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = kernel_size,
                            padding = kernel_size // 2,
                            bias = False,
                            ),
                    ),
                    ('norm1', nn.BatchNorm2d(num_features=features)),
                    ('lrelu1', nn.LeakyReLU()),
                    (
                        'conv2',
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = kernel_size,
                            padding = kernel_size // 2,
                            bias = False,
                            ),
                    ),
                    ('norm2', nn.BatchNorm2d(num_features=features)),
                    ('lrelu2', nn.LeakyReLU()),
                ]
            )
        )


class Decoder(nn.Module):
    def __init__(self,
            out_channels = 1,
            features = 128,
            in_shape = (64, 64),
            ):
        super(Decoder, self).__init__()

        self.in_shape = in_shape
        self.features = features

        self.fc = nn.Linear(
            in_features = features, 
            out_features = (in_shape[0] // 8) * (in_shape[1] // 8) * features,
            )

        self.up1 = Decoder._upblock(features, features)
        self.decoder1 = Decoder._block(features, features//2)
        self.up2 = Decoder._upblock(features//2, features//2)
        self.decoder2 = Decoder._block(features//2, features//4)
        self.up3 = Decoder._upblock(features//4, features//4)

        self.out = nn.Conv2d(in_channels=features//4, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc(x).view(-1, self.features, self.in_shape[0]//8, self.in_shape[1]//8)
        x = self.decoder1(self.up1(x))
        x = self.decoder2(self.up2(x))
        x = self.up3(x)

        return self.out(x)

    @staticmethod
    def _upblock(
            in_channels,
            features,
            kernel_size = 2,
            stride = 2,
            ):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        'convT',
                        nn.ConvTranspose2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = kernel_size,
                            stride = stride,
                            ),
                    ),
                    ('lrelu', nn.LeakyReLU()),
                ]
            )
        )
        
    @staticmethod
    def _block(
            in_channels,
            features,
            kernel_size = 3,
            ):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        'conv1',
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = kernel_size,
                            padding = kernel_size // 2,
                            bias = False,
                            ),
                    ),
                    ('norm1', nn.BatchNorm2d(num_features=features)),
                    ('lrelu1', nn.LeakyReLU()),
                    (
                        'conv2',
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = kernel_size,
                            padding = kernel_size // 2,
                            bias = False,
                            ),
                    ),
                    ('norm2', nn.BatchNorm2d(num_features=features)),
                    ('lrelu2', nn.LeakyReLU()),
                ]
            )
        )
