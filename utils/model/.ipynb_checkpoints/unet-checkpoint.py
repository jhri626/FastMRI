"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        assert not torch.isnan(image).any(), "NaN in input image"
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            assert not torch.isnan(output).any(), "NaN after down_sample layer"
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            assert not torch.isnan(output).any(), "NaN after avg_pool2d"


        output = self.conv(output)
        assert not torch.isnan(output).any(), "NaN after conv"

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            assert not torch.isnan(output).any(), "NaN after transpose_conv"

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
                assert not torch.isnan(output).any(), "NaN after F.pad"


            output = torch.cat([output, downsample_layer], dim=1)
            assert not torch.isnan(output).any(), "NaN after torch.cat"
            output = conv(output)
            assert not torch.isnan(output).any(), "NaN after up_conv"
        
        return output


"""
Facebook Unet Layers
    ConvBlock
    TransposeConvBlock
"""


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )
        #self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(layer, nn.InstanceNorm2d):
                if layer.weight is not None:
                    nn.init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        #return self.layers(image)
        #self.check_for_nan_in_parameters()
        
        assert not torch.isnan(image).any(), "NaN in input data of ConvBlock"

        print(f"ConvBlock weights after second Conv2d: {self.layers[0].weight.data}")
        output = self.layers[0](image)
        #assert not torch.isnan(output).any(), "NaN after first Conv2d"

        output = self.layers[1](output)
        assert not torch.isnan(output).any(), "NaN after first InstanceNorm2d"

        output = self.layers[2](output)
        assert not torch.isnan(output).any(), "NaN after first LeakyReLU"

        output = self.layers[3](output)
        assert not torch.isnan(output).any(), "NaN after first Dropout2d"

        output = self.layers[4](output)
        assert not torch.isnan(output).any(), "NaN after second Conv2d"

        output = self.layers[5](output)
        assert not torch.isnan(output).any(), "NaN after second InstanceNorm2d"

        output = self.layers[6](output)
        assert not torch.isnan(output).any(), "NaN after second LeakyReLU"

        output = self.layers[7](output)
        assert not torch.isnan(output).any(), "NaN after second Dropout2d"

        return output
        #output=self.layers(image)
        #assert not torch.isnan(output).any(), "NaN in ConvBlock"
        #return output
        
    def check_for_nan_in_parameters(self):
        """
        ConvBlock의 모든 파라미터에서 NaN 값을 검사합니다.
        NaN이 발견되면 AssertionError를 발생시킵니다.
        """
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise AssertionError(f"NaN detected in parameter: {name}")


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        #return self.layers(image)
        output = self.layers(image)
        assert not torch.isnan(output).any(), "NaN in TransposeConvBlock"
        return output