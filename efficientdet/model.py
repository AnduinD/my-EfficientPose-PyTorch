import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as nms_torch

from efficientnet import EfficientNet as EffNet
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

from utils.utils import gather_torch, gather_nd_simple, gather_nd_batch


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.parameter.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.parameter.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.parameter.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.parameter.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.parameter.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.parameter.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.parameter.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.parameter.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4) # type: ignore
            p5_in = self.p5_down_channel_2(p5) # type: ignore

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in))) # type: ignore

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4) # type: ignore
            p5_in = self.p5_down_channel_2(p5) # type: ignore

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out))) # type: ignore

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out))) # type: ignore

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list): #type: ignore
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        return feats

class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list): #type: ignore
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], 
                                          feat.shape[1], 
                                          feat.shape[2], 
                                          self.num_anchors, 
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        # print("in classfier")
        # print(len(feats))
        # print(feats[0].shape)
        # print(feats[1].shape)
        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats

class IterativeTranslationSubNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_iter_steps, pyramid_levels=5,num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = 2, **kwargs):
        super(IterativeTranslationSubNet, self).__init__(**kwargs)
        # 多个尺度和多个迭代次数里的细化模块，用的是同一套conv参数，但是各自的norm参数独立
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_anchors = num_anchors
        self.num_iter_steps = num_iter_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.pyramid_levels = pyramid_levels
       
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock( in_channels = in_channels if i==0 else out_channels,
                                 out_channels = out_channels, 
                                 norm=False, activation=False) 
                                 for i in range(num_layers)])
        self.head_xy = SeparableConvBlock(in_channels = self.out_channels,
                                          out_channels = self.num_anchors * 2,
                                          norm=False, activation=False)
        self.head_z = SeparableConvBlock(in_channels = self.out_channels, 
                                         out_channels = self.num_anchors,
                                         norm=False, activation=False)
        
        if self.use_group_norm:
            self.norm_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([
                nn.GroupNorm(self.num_groups_gn, out_channels) 
                for i in range(num_layers)]) 
                for j in range(self.num_iter_steps)])
                for k in range(pyramid_levels)])
        else: 
            self.norm_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.003, eps=1e-4) 
                for i in range(num_layers)]) 
                for j in range(self.num_iter_steps)])
                for k in range(pyramid_levels)])

        self.activation = MemoryEfficientSwish() #if not onnx_export else Swish()

    def forward(self, inputs, cur_level, cur_iter_step):
        #for feat, level_norm_list in zip(inputs, self.norm_list): # 循环多尺度level
        feat = inputs
        
        for cur_layer, conv in zip(range(self.num_layers) ,self.conv_list):  
            # 循环一个细化模块里的layer层数
            feat = conv(feat)
            feat = self.norm_list[cur_level][cur_iter_step][cur_layer](feat)  #type: ignore
                        # 因为GN层不共享 所以索引起来有点怪
            feat = self.activation(feat)
        delta_xy = self.head_xy(feat)
        delta_z = self.head_z(feat)    
        return delta_xy,delta_z

class TranslationNet(nn.Module):
    """
    modified by Anduin
    Args:
        in_channels/subnet_width: The number of channels used in the subnetwork layers
        num_layers/subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    """

    def __init__(self, in_channels, num_layers, num_iter_steps, pyramid_levels=5, num_anchors=9,onnx_export=False, freeze_bn = False, use_group_norm = True, num_groups_gn = 2):
        super(TranslationNet, self).__init__()
        self.num_layers = num_layers
        self.in_channels=in_channels
        self.num_anchors=num_anchors
        self.num_iter_steps = num_iter_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.pyramid_levels = pyramid_levels

        # 子网络的分离卷积和初始回归
        self.conv_list = nn.ModuleList([
            SeparableConvBlock(in_channels = self.in_channels,
                               out_channels = self.in_channels, 
                               norm=False, activation=False) 
            for i in range(num_layers)])
        self.initial_translation_xy = SeparableConvBlock(in_channels = in_channels,
                                                         out_channels = self.num_anchors * 2, 
                                                         norm=False, activation=False)
        self.initial_translation_z = SeparableConvBlock(in_channels = in_channels,
                                                        out_channels = self.num_anchors, 
                                                        norm=False, activation=False)
        
        # 子网络的norm
        if self.use_group_norm:
            self.norm_list = nn.ModuleList([nn.ModuleList([
                nn.GroupNorm(self.num_groups_gn, in_channels) 
                for i in range(num_layers)]) 
                for j in range(pyramid_levels)])
        else: 
            self.norm_list = nn.ModuleList([nn.ModuleList([
                nn.BatchNorm2d(in_channels, momentum=0.003, eps=1e-4) 
                for i in range(num_layers)]) 
                for j in range(pyramid_levels)])

        # 子网络的迭代细化块
        self.iter_submodel = IterativeTranslationSubNet(in_channels= self.in_channels+self.num_anchors * 2+self.num_anchors,
                                                        out_channels=self.in_channels,
                                                             num_layers=self.num_layers - 1,
                                                             num_iter_steps = self.num_iter_steps,
                                                             num_anchors = self.num_anchors,
                                                             freeze_bn = freeze_bn,
                                                             use_group_norm= self.use_group_norm,
                                                             num_groups_gn = self.num_groups_gn,
                                                             pyramid_levels= pyramid_levels)

        self.activation = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.level = 0

    def forward(self, inputs):
        feats = []
        #print(f"translationnet input shape: {inputs[0].shape}")
        for cur_level, feat, norm_list in zip(range(self.pyramid_levels), inputs, self.norm_list): 
            # 循环多尺度level
            for cur_layer, norm, conv in zip(range(self.num_layers), norm_list, self.conv_list):  # type: ignore
                # 循环初始回归的层数
                feat = conv(feat)
                feat = norm(feat)
                feat = self.activation(feat)
            
            trans_xy = self.initial_translation_xy(feat)
            trans_z = self.initial_translation_z(feat)

            for cur_iter_step in range(self.num_iter_steps): #循环迭代细化的层数
                iter_input = torch.cat((feat, trans_xy, trans_z),dim=1) #在C维叠加 tf1源码里的通道在最后一维
                delta_trans_xy, delta_trans_z = self.iter_submodel(iter_input,cur_level,cur_iter_step) 
                trans_xy = trans_xy + delta_trans_xy 
                trans_z = trans_z + delta_trans_z 

            outputs_xy = trans_xy.reshape((trans_xy.shape[0],-1,2))
            outputs_z = trans_z.reshape((trans_xy.shape[0],-1,1))
            outputs = torch.cat((outputs_xy,outputs_z),dim=2)
            feats.append(outputs) # 将单层level的输出加到多尺度输出的表里
       
        # print("in translation")
        # print(len(feats))
        # print(feats[0].shape)
        # print(feats[1].shape)
        feats = torch.cat(feats, dim=1)
        #feats = feats.sigmoid()

        return feats  # 多尺度

class IterativeRotationSubNet(nn.Module):
    def __init__(self, in_channels,out_channels, num_layers, num_rot_params, num_iter_steps, pyramid_levels=5, num_anchors = 9, freeze_bn = False, use_group_norm = True, num_groups_gn = 2, **kwargs):
        super(IterativeRotationSubNet, self).__init__(**kwargs)
        # 多个尺度和多个迭代次数里的细化模块，用的是同一套conv参数，但是各自的norm参数独立
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_anchors = num_anchors
        self.num_iter_steps = num_iter_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.pyramid_levels = pyramid_levels
        self.num_rot_params = num_rot_params

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels = in_channels if i==0 else out_channels,
                                 out_channels = out_channels, 
                                 norm=False, activation=False) 
                                 for i in range(num_layers)])
        self.head_rot = SeparableConvBlock(in_channels=self.out_channels,
                                           out_channels = self.num_anchors * self.num_rot_params,
                                           norm=False, activation=False)

        if self.use_group_norm:
            self.norm_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([
                nn.GroupNorm(self.num_groups_gn, self.out_channels) 
                for i in range(num_layers)]) 
                for j in range(self.num_iter_steps)])
                for k in range(pyramid_levels)])
        else: 
            self.norm_list = nn.ModuleList([nn.ModuleList([nn.ModuleList([
                nn.BatchNorm2d(self.out_channels, momentum=0.003, eps=1e-4) 
                for i in range(num_layers)]) 
                for j in range(self.num_iter_steps)])
                for k in range(pyramid_levels)])

        self.activation = MemoryEfficientSwish() #if not onnx_export else Swish()

    def forward(self, inputs, cur_level, cur_iter_step):
        #for feat, level_norm_list in zip(inputs, self.norm_list): # 循环多尺度level
        feat = inputs
        for cur_layer, conv in zip(range(self.num_layers) ,self.conv_list):  
            # 循环一个细化模块里的layer层数
            feat = conv(feat)
            feat = self.norm_list[cur_level][cur_iter_step][cur_layer](feat)  # type: ignore
            # 因为GN层不共享 所以索引起来有点怪
            feat = self.activation(feat)
        delta_rot = self.head_rot(feat)
        return delta_rot
    
class RotationNet(nn.Module):
    """
    modified by Anduin
    Args:
        in_channels/subnet_width: The number of channels used in the subnetwork layers
        num_layers/subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    """
    def __init__(self, in_channels, num_anchors, num_layers, num_rot_params, num_iter_steps, pyramid_levels=5, onnx_export=False, freeze_bn = False, use_group_norm = True, num_groups_gn = 2):
        super(RotationNet, self).__init__()
        self.num_layers = num_layers
        self.in_channels=in_channels
        self.num_anchors=num_anchors
        self.num_iter_steps = num_iter_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        self.pyramid_levels = pyramid_levels
        self.num_rot_params = num_rot_params

        # 子网络的分离卷积和初始回归
        self.conv_list = nn.ModuleList([
            SeparableConvBlock(in_channels = self.in_channels,
                               out_channels = self.in_channels, 
                               norm=False, activation=False) 
            for i in range(num_layers)])
        self.initial_rot = SeparableConvBlock(in_channels = self.in_channels,
                                              out_channels = self.num_anchors * self.num_rot_params,
                                              norm=False, activation=False)

        # 子网络的norm
        if self.use_group_norm:
            self.norm_list = nn.ModuleList([nn.ModuleList([
                nn.GroupNorm(self.num_groups_gn, in_channels) 
                for i in range(num_layers)]) 
                for j in range(pyramid_levels)])
        else: 
            self.norm_list = nn.ModuleList([nn.ModuleList([
                nn.BatchNorm2d(in_channels, momentum=0.003, eps=1e-4) 
                for i in range(num_layers)]) 
                for j in range(pyramid_levels)])
        
        # 子网络的迭代细化块
        self.iter_submodel = IterativeRotationSubNet(
                            in_channels= self.in_channels+self.num_anchors * self.num_rot_params,
                            out_channels= self.in_channels,
                            num_layers=self.num_layers - 1,
                            num_rot_params=self.num_rot_params,
                            num_iter_steps = self.num_iter_steps,
                            num_anchors = self.num_anchors,
                            freeze_bn = freeze_bn,
                            use_group_norm= self.use_group_norm,
                            num_groups_gn = self.num_groups_gn,
                            pyramid_levels= pyramid_levels)

        self.activation = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        feat_img = inputs[0]
        feat_cam = inputs[1]
        for cur_level, feat, norm_list in zip(range(self.pyramid_levels), feat_img, self.norm_list): 
            # 循环多尺度level
            for cur_layer, norm, conv in zip(range(self.num_layers), norm_list, self.conv_list):  # type: ignore
                # 循环初始回归的层数
                feat = conv(feat)
                feat = norm(feat)
                feat = self.activation(feat)
            
            rot = self.initial_rot(feat)

            for cur_iter_step in range(self.num_iter_steps): #循环迭代细化的层数
                iter_input = torch.cat((feat, rot),dim=1)
                delta_rot = self.iter_submodel(iter_input,cur_level,cur_iter_step) 
                rot = rot + delta_rot 

            outputs_rot = rot.reshape((rot.shape[0],-1,self.num_rot_params))
            feats.append(outputs_rot) # 将单层level的输出加到多尺度输出的表里
        
        feats = torch.cat(feats, dim=1)
        #feats = feats.sigmoid()

        return feats  # 多尺度

class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model
        #print(f"EffcientNet model:{model}")

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate # type: ignore
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]



def filter_detections(
        boxes,
        classification,
        rotation,
        translation,
        num_rotation_parameters,
        num_translation_parameters = 3,
        class_specific_filter = True,
        nms = True,
        score_threshold = 0.01,
        max_detections = 100,
        nms_threshold = 0.5,
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        rotation: Tensor of shape (num_boxes, num_rotation_parameters) containing the rotations.
        translation: Tensor of shape (num_boxes, 3) containing the translation vectors.
        num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
        num_translation_parameters: Number of translation parameters, usually 3 
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, rotation, translation].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        rotation is shaped (max_detections, num_rotation_parameters) and contains the rotations of the non-suppressed predictions.
        translation is shaped (max_detections, num_translation_parameters) and contains the translations of the non-suppressed predictions.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = torch.nonzero(torch.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = gather_nd_simple(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = gather_torch(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
            # nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
            #                                            iou_threshold=nms_threshold)
            nms_indices = nms_torch(boxes = filtered_boxes, 
                                    scores = filtered_scores, 
                                    iou_threshold=nms_threshold)[:max_detections]
            #nms_indices = nms_indices[:max_detections]


            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            print(f"nms_indices.shape: {nms_indices.shape}")
            indices_ = gather_torch(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = gather_nd_simple(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = torch.stack([indices_[:, 0], labels_], dim=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * torch.ones(((scores.shape)[0],), dtype=torch.int64)
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = torch.cat(all_indices, dim=0)
    else:
        scores, labels = torch.max(classification, dim=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = gather_nd_simple(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = torch.topk(scores, k = (scores.shape)[0] if max_detections > (scores.shape)[0]  else max_detections )

    # filter input using the final set of indices
    indices = gather_torch(indices[:, 0], top_indices)
    boxes = gather_torch(boxes, indices)
    labels = gather_torch(labels, top_indices)
    rotation = gather_torch(rotation, indices)
    translation = gather_torch(translation, indices)

    # zero pad the outputs
    pad_size = max_detections - (scores.shape)[0] if max_detections - (scores.shape)[0] > 0 else 0
    boxes = F.pad(boxes, (0, pad_size, 0, 0), value=-1)
    scores = F.pad(scores, (0, pad_size), value=-1)
    labels = F.pad(labels, (0, pad_size), value=-1)
    labels = labels.to('int32')
    rotation = F.pad(rotation, (0, pad_size, 0, 0), value=-1)
    translation = F.pad(translation, (0, pad_size, 0, 0), value=-1)

    # set shapes, since we know what they are
    boxes = boxes.view([max_detections, 4])
    scores = scores.view([max_detections])
    labels = labels.view([max_detections])
    rotation = rotation.view([max_detections, num_rotation_parameters])
    translation = translation.view([max_detections, num_translation_parameters])

    return [boxes, scores, labels, rotation, translation]


class FilterDetections(nn.Module):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            num_rotation_parameters,
            num_translation_parameters = 3,
            nms = True,
            class_specific_filter = True,
            nms_threshold = 0.5, 
            score_threshold = 0.01,
            max_detections = 100,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
            num_translation_parameters: Number of translation parameters, usually 3 
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        super(FilterDetections, self).__init__(**kwargs)

    def forward(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, rotation, translation] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        translation = inputs[2]
        rotation = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            rotation_ = args[2]
            translation_ = args[3]

            return filter_detections(
                boxes_,
                classification_,
                rotation_,
                translation_,
                self.num_rotation_parameters,
                self.num_translation_parameters,
                nms = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold = self.score_threshold,
                max_detections = self.max_detections,
                nms_threshold = self.nms_threshold,
            )

        # call filter_detections on each batch item
        # 这一段处理肯定有问题  outputs应该是对每个filter_对象的append
        outputs = torch.Tensor([])
        for i in range((boxes.shape)[0]):
            assert (boxes[i].shape)[0] == (classification[i].shape)[0]
            assert (boxes[i].shape)[0] == (rotation[i].shape)[0]
            assert (boxes[i].shape)[0] == (translation[i].shape)[0]
            outputs = torch.cat((outputs, _filter_detections([boxes[i], classification[i], rotation[i], translation[i]])), dim = 0) # type: ignore

            # if i == 0:
            #     filtered_boxes = outputs[0]  
            #     filtered_scores = outputs[1]
            #     filtered_labels = outputs[2]
            #     filtered_rotation = outputs[3]
            #     filtered_translation = outputs[4]
            # else:
            #     filtered_boxes = torch.cat((filtered_boxes, outputs[0]), 0) #type: ignore
            #     filtered_scores = torch.cat((filtered_scores, outputs[1]), 0) #type: ignore
            #     filtered_labels = torch.cat((filtered_labels, outputs[2]), 0) #type: ignore
            #     filtered_rotation = torch.cat((filtered_rotation, outputs[3]), 0) #type: ignore
            #     filtered_translation = torch.cat((filtered_translation, outputs[4]), 0) #type: ignore

        # outputs = tf.map_fn(
        #     _filter_detections,
        #     elems=[boxes, classification, rotation, translation],
        #     dtype=['float32', 'float32', 'int32', 'float32', 'float32'],
        #     parallel_iterations=self.parallel_iterations
        # )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, rotation, translation].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_rotation.shape, filtered_translation.shape]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[2][0], self.max_detections, self.num_rotation_parameters),
            (input_shape[3][0], self.max_detections, self.num_translation_parameters),
        ]

    def compute_mask(self, inputs, mask = None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config() # type: ignore
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
            'num_rotation_parameters': self.num_rotation_parameters,
            'num_translation_parameters': self.num_translation_parameters,
        })

        return config
    
    

if __name__ == '__main__':
    #from tensorboard import SummaryWriter

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
