from typing import Optional, List

import mindspore.nn as nn
import mindspore.ops as ops

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.pooling import GlobalAvgPooling
from .utils import make_divisible, load_pretrained
from .registry import register_model

'''
1. 基本结构和timm都是一样的，没有太大差别, 但是
网络最后的分类器pytorch是直接用全连接层做分类器，这里是参考论文和tf的用法，直接用卷积层做分类器，
正好也已经有训练好的22个规模的预训练模型, 精度都达标
2. 同样将 conv2d+bn+relu 改成了Conv2dNormActivation
3. 全局平均池化使用提出来公用的GlobalAvgPooling
'''

__all__ = ['InvertedResidual', 'MobileNetV2']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1001,
        'first_conv': 'features.0.features.0', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenet_v2_1.4_224': _cfg(url=''),
    'mobilenet_v2_1.3_224': _cfg(url=''),
    'mobilenet_v2_1.0_224': _cfg(url=''),
    'mobilenet_v2_1.0_192': _cfg(url=''),
    'mobilenet_v2_1.0_160': _cfg(url=''),
    'mobilenet_v2_1.0_128': _cfg(url=''),
    'mobilenet_v2_1.0_96': _cfg(url=''),
    'mobilenet_v2_0.75_224': _cfg(url=''),
    'mobilenet_v2_0.75_192': _cfg(url=''),
    'mobilenet_v2_0.75_160': _cfg(url=''),
    'mobilenet_v2_0.75_128': _cfg(url=''),
    'mobilenet_v2_0.75_96': _cfg(url=''),
    'mobilenet_v2_0.5_224': _cfg(url=''),
    'mobilenet_v2_0.5_192': _cfg(url=''),
    'mobilenet_v2_0.5_160': _cfg(url=''),
    'mobilenet_v2_0.5_128': _cfg(url=''),
    'mobilenet_v2_0.5_96': _cfg(url=''),
    'mobilenet_v2_0.35_224': _cfg(url=''),
    'mobilenet_v2_0.35_192': _cfg(url=''),
    'mobilenet_v2_0.35_160': _cfg(url=''),
    'mobilenet_v2_0.35_128': _cfg(url=''),
    'mobilenet_v2_0.35_96': _cfg(url=''),
}


class InvertedResidual(nn.Cell):

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int,
                 expand_ratio: int,
                 norm: Optional[nn.Cell] = None,
                 last_relu: bool = False
                 ) -> None:
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        if not norm:
            norm = nn.BatchNorm2d

        hidden_dim = round(in_channel * expand_ratio)
        self.use_res_connect = stride == 1 and in_channel == out_channel

        layers: List[nn.Cell] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(in_channel, hidden_dim, kernel_size=1, norm=norm, activation=nn.ReLU6)
            )
        layers.extend([
            # dw
            Conv2dNormActivation(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm=norm,
                activation=nn.ReLU6
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channel, kernel_size=1,
                      stride=1, has_bias=False),
            norm(out_channel)
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = ops.Add()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = self.add(identity, x)
        if self.last_relu:
            x = self.relu(x)
        return x


class MobileNetV2(nn.Cell):

    def __init__(self,
                 alpha: float = 1.0,
                 inverted_residual_setting: Optional[List[List[int]]] = None,
                 round_nearest: int = 8,
                 block: Optional[nn.Cell] = None,
                 norm: Optional[nn.Cell] = None,
                 num_classes: int = 1000,
                 in_channels: int = 3
                 ) -> None:
        super(MobileNetV2, self).__init__()

        if not block:
            block = InvertedResidual
        if not norm:
            norm = nn.BatchNorm2d

        input_channel = make_divisible(32 * alpha, round_nearest)
        self.last_channel = make_divisible(1280 * max(1.0, alpha), round_nearest)

        # Setting of inverted residual blocks.
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Building first layer.
        features: List[nn.Cell] = [
            Conv2dNormActivation(in_channels, input_channel, stride=2, norm=norm, activation=nn.ReLU6)
        ]

        # Building inverted residual blocks.
        # t: The expansion factor.
        # c: Number of output channel.
        # n: Number of block.
        # s: First block stride.
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm=norm))
                input_channel = output_channel

        # Building last several layers.
        features.append(
            Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, norm=norm, activation=nn.ReLU6)
        )
        # Make it nn.CellList.
        self.features = nn.SequentialCell(features)

        self.pool = GlobalAvgPooling(keep_dims=True)

        self.classifier = nn.Conv2d(self.last_channel, num_classes, kernel_size=1, has_bias=True)
        self.flatten = nn.Flatten()

    # 获取分类器的部分， 实现参考了Timmy的get_classifier的写法
    def get_classifier(self):
        return self.classifier

    # 修改分类器，参考了Timmy的写法
    def reset_classifier(self, num_classes):
        self.classifier = nn.Conv2d(self.last_channel, num_classes, kernel_size=1, has_bias=True)

    # 该函数主要是特征提取层，获取特征的，实现参考了Timmy的forwar_features的写法
    def get_features(self, x):
        x = self.features(x)
        return x

    def construct(self, x):
        x = self.get_features(x)
        x = self.pool(x)
        x = self.classifier(x)
        x = self.flatten(x)

        return x


@register_model
def mobilenet_v2_140_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.4_224']
    model = MobileNetV2(alpha=1.4, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_130_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.3_224']
    model = MobileNetV2(alpha=1.3, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.0_224']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.0_192']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.0_160']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.0_128']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['mobilenet_v2_1.0_96']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
