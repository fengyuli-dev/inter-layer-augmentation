from torch import nn
import augmentations


class ResNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=True,
        activation=True,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if activation:
            self.conv_layers.append(nn.ReLU())

    def forward(self, hidden_state):
        return self.conv_layers(hidden_state)


class ResiduleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        first_stride = 2 if downsample else 1
        self.conv_layers = nn.Sequential(
            ResNetConvLayer(in_channels, out_channels, stride=first_stride),
            ResNetConvLayer(out_channels, out_channels, activation=False),
        )
        self.skip_connection = nn.Sequential(
            ResNetConvLayer(
                in_channels,
                out_channels,
                stride=first_stride,
                bias=False,
                activation=False,
            ),
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.conv_layers(hidden_state)
        residual = self.skip_connection(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class BottleneckBlock(nn.Module):
    def __init__(
        self, in_channels, bottlenecked_channels, out_channels, downsample=False
    ):
        super().__init__()
        first_stride = 2 if downsample else 1
        self.conv_layers = nn.Sequential(
            ResNetConvLayer(
                in_channels, bottlenecked_channels, kernel_size=1, stride=first_stride
            ),
            ResNetConvLayer(bottlenecked_channels, bottlenecked_channels),
            ResNetConvLayer(
                bottlenecked_channels, out_channels, kernel_size=1, activation=False
            ),
        )
        self.skip_connection = nn.Sequential(
            ResNetConvLayer(
                in_channels,
                out_channels,
                stride=first_stride,
                bias=False,
                activation=False,
            ),
        )

        self.activation = nn.ReLU()

    def forward(self, hidden_state):
        residule = hidden_state
        hidden_state = self.conv_layers(hidden_state)
        hidden_state += self.skip_connection(residule)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        is_imagenet = num_classes == 1000
        self.first_conv = ResNetConvLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=7 if is_imagenet else 3,
            stride=2 if is_imagenet else 1,
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            if is_imagenet
            else nn.Identity()
        )
        channels_config = [
            (64, 64, 2),
            (64, 256, 2),
            (256, 512, 2),
        ]
        resnet_blocks_list = nn.ModuleList([])
        for i, (
            in_channels,
            out_channels,
            num_blocks,
        ) in enumerate(channels_config):
            resnet_blocks_list.append(
                # No downsample for the first block as we have already downsampled in the first conv layer
                ResiduleBlock(in_channels, out_channels, downsample=i != 0)
            )
            # resnet_blocks_list.append(augmentations.GridDropout())
            for _ in range(num_blocks - 1):
                resnet_blocks_list.append(ResiduleBlock(out_channels, out_channels))
        self.resnet_blocks = nn.Sequential(*resnet_blocks_list)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, pixel_values):
        hidden_state = self.first_conv(pixel_values)
        hidden_state = self.max_pool(hidden_state)
        hidden_state = self.resnet_blocks(hidden_state)
        hidden_state = self.avg_pool(hidden_state)
        hidden_state = hidden_state.view(hidden_state.size(0), -1)
        logits = self.fc(hidden_state)
        return logits


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        is_imagenet = num_classes == 1000
        self.first_conv = ResNetConvLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=7 if is_imagenet else 3,
            stride=2 if is_imagenet else 1,
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            if is_imagenet
            else nn.Identity()
        )
        channels_config = [
            (64, 64, 256, 3),
            (256, 128, 512, 4),
            (512, 256, 1024, 6),
            (1024, 512, 2048, 3),
        ]
        resnet_blocks_list = nn.ModuleList([])
        for i, (
            in_channels,
            bottlenecked_channels,
            out_channels,
            num_blocks,
        ) in enumerate(channels_config):
            resnet_blocks_list.append(
                # No downsample for the first block as we have already downsampled in the first conv layer
                BottleneckBlock(
                    in_channels, bottlenecked_channels, out_channels, downsample=i != 0
                )
            )
            for _ in range(num_blocks - 1):
                resnet_blocks_list.append(
                    BottleneckBlock(out_channels, bottlenecked_channels, out_channels)
                )
        self.resnet_blocks = nn.Sequential(*resnet_blocks_list)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, pixel_values):
        hidden_state = self.first_conv(pixel_values)
        hidden_state = self.max_pool(hidden_state)
        hidden_state = self.resnet_blocks(hidden_state)
        hidden_state = self.avg_pool(hidden_state)
        hidden_state = hidden_state.view(hidden_state.size(0), -1)
        logits = self.fc(hidden_state)
        return logits
