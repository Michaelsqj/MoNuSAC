import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch import nn
from torch.nn import functional as F
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class _BNRelu(nn.Module):
    def __init__(self, num_features):
        super(_BNRelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, inputs):
        return F.relu(self.bn(inputs), inplace=True)


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                 expansion=4, preact=True):
        super(_ResidualUnit, self).__init__()
        self.preact = preact
        bottleneck_channels = out_channels // expansion

        self.bn_relu1 = _BNRelu(in_channels)

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, dilation=1, bias=False)
        self.bn_relu2 = _BNRelu(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               dilation=1, bias=False)
        self.bn_relu3 = _BNRelu(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, dilation=1, bias=False)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, padding=0, dilation=1,
                                      bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        out = self.bn_relu1(inputs) if self.preact else inputs
        shortcut = self.shortcut(inputs)
        out = self.bn_relu2(self.conv1(out))
        out = self.bn_relu3(self.conv2(out))
        out = self.conv3(out)
        out += shortcut

        return out


class _DenseUnit(nn.Module):
    def __init__(self, in_channels):
        super(_DenseUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.bn_relu1 = _BNRelu(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=5,
                               padding=2, stride=1, dilation=1, bias=False)
        self.bn_relu2 = _BNRelu(32)

    def forward(self, inputs):
        out = self.bn_relu1(self.conv1(inputs))
        out = self.bn_relu2(self.conv2(out))

        return torch.cat([out, inputs], dim=1)


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               padding=3, stride=1, dilation=1, bias=False)
        self.residual_block1 = nn.Sequential(
            _ResidualUnit(64, 256, stride=1, preact=False)
        )
        self.residual_block2 = nn.Sequential(
            _ResidualUnit(256, 512, stride=2, preact=False),
            _ResidualUnit(512, 512, stride=1, preact=True)
        )
        self.residual_block3 = nn.Sequential(
            _ResidualUnit(512, 1024, stride=2, preact=False),
            _ResidualUnit(1024, 1024, stride=1, preact=True),
            _ResidualUnit(1024, 1024, stride=1, preact=True)
        )
        self.residual_block4 = nn.Sequential(
            _ResidualUnit(1024, 2048, stride=2, preact=False)
        )
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.conv2(x)

        return x


# TODO Remove hardcoded layers
class _Decoder(nn.Module):
    def __init__(self, input_shape, in_channels):
        super(_Decoder, self).__init__()
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=5,
                               padding=2, stride=1, dilation=1, bias=False)
        self.dense_block1 = nn.Sequential(
            _DenseUnit(256),
            _DenseUnit(256 + 32 * 1),
            _DenseUnit(256 + 32 * 2),
        )
        self.conv2 = nn.Conv2d(256 + 32 * 3, 512, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=5, padding=2,
                               bias=False)
        self.dense_block2 = nn.Sequential(
            _DenseUnit(128),
            _DenseUnit(128 + 32)
        )
        self.conv4 = nn.Conv2d(128 + 32 * 2, 128, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5,
                               padding=2, stride=1, dilation=1, bias=False)
        self.conv6 = nn.Conv2d(256, 64, kernel_size=1,
                               padding=0, stride=1, dilation=1, bias=False)

    def forward(self, inputs):
        # TODO: Replace interpolate with 2x2 unpooling
        x = F.interpolate(inputs, scale_factor=2)
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.conv3(F.interpolate(self.conv2(x), scale_factor=2))
        x = self.dense_block2(x)
        x = self.conv5(F.interpolate(self.conv4(x), scale_factor=2))
        x = self.conv6(x)
        # x N*64*256*256
        return x


class _SegmentationHead(nn.Module):
    def __init__(self, head):
        super(_SegmentationHead, self).__init__()
        assert head in ['np', 'hv', 'nc']  # "Head must be 'np' or 'hv' or 'nc"
        self.head = head

        self.bn_relu = _BNRelu(num_features=64)
        self.conv1 = nn.Conv2d(64, 2, kernel_size=1, padding=0, stride=1, dilation=1, bias=True)
        self.conv3 = nn.Conv2d(64, 5, kernel_size=1, padding=0, stride=1, dilation=1, bias=True)

    def forward(self, inputs):
        out = self.bn_relu(inputs)
        if self.head == 'np':
            out = self.conv1(out)
            assert out.shape[1] == 2
        elif self.head == 'nc':
            out = self.conv3(out)
            assert out.shape[1] == 5
        assert out.dim() == 4
        return out


class HoverNet(nn.Module):
    def __init__(self):
        super(HoverNet, self).__init__()
        self.encoder = _Encoder()
        self.decoder_np = _Decoder((256, 256), 1024)
        self.decoder_nc = _Decoder((256, 256), 1024)

        self.head_np = _SegmentationHead(head='np')
        self.head_nc = _SegmentationHead(head='nc')

    def forward(self, inputs):
        # assert inputs.shape[1] == 3
        x = self.encoder(inputs)

        np = self.decoder_np(x)
        nc = self.decoder_nc(x)

        out_np = self.head_np(np)
        out_nc = self.head_nc(nc)


        return out_np, out_nc


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = HoverNet()
    model = model.to(device)
    # summary(model, (3, 256, 256))
    writer = SummaryWriter('/home/jqshen/MyCode/MyModel/tensorboard')
    writer.add_graph(model, torch.randn(size=(2, 3, 256, 256)).to(device))
    writer.close()
