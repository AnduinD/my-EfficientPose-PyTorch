# Author: Zylo117

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier,TranslationNet,RotationNet, EfficientNet
from efficientdet.utils import Anchors


class EfficientPoseBackbone(nn.Module):
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        score_threshold = 0.5, # nms后处理用到的
                        anchor_parameters = None, # translation后处理用到的
                        num_rotation_parameters = 3,
                        **kwargs):
        super(EfficientPoseBackbone, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.bifpn_widths = [64, 88, 112, 160, 224, 288, 384, 384, 384] #fpn_num_filters
        self.bifpn_depths = [3, 4, 5, 6, 7, 7, 8, 8, 8] #fpn_cell_repeats
        self.subnet_widths = self.bifpn_widths
        self.subnet_depths = [3, 3, 3, 4, 4, 4, 5, 5, 5] #box_class_repeats
        self.subnet_iter_steps = [1, 1, 1, 2, 2, 2, 3, 0, 0] # rot和trans的细化迭代次数
        self.num_groups_gn = [4, 4, 7, 10, 14, 18, 24, 0, 0] #try to get 16 channels per group
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6] #bifpn输出的金字塔层数（不同尺度）
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.num_rot_params = num_rotation_parameters
        
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales
        # print(f"num_anchors:{num_anchors}")

        self.bifpn = nn.Sequential(
            *[BiFPN(self.bifpn_widths[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False, 
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.bifpn_depths[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.subnet_widths[self.compound_coef], 
                                   num_anchors=num_anchors,
                                   num_layers=self.subnet_depths[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = Classifier(in_channels=self.subnet_widths[self.compound_coef], 
                                     num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.subnet_depths[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.translation = TranslationNet(in_channels=self.subnet_widths[self.compound_coef], 
                                          num_anchors=num_anchors,
                                          num_layers=self.subnet_depths[self.compound_coef],
                                          num_iter_steps=self.subnet_iter_steps[self.compound_coef],
                                          pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.rotation = RotationNet(in_channels=self.subnet_widths[self.compound_coef], 
                                    num_anchors=num_anchors,
                                    num_layers=self.subnet_depths[self.compound_coef],
                                    num_rot_params=self.num_rot_params,
                                    num_iter_steps=self.subnet_iter_steps[self.compound_coef],
                                    pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        input_img, input_cam = inputs
        max_size = input_img.shape[-1]

        _, p3, p4, p5 = self.backbone_net(input_img)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        translation  = self.translation(features)
        rotation = self.rotation([features,input_cam])
        
        anchors = self.anchors(input_img, input_img.dtype)


        
        return features, regression, classification, translation, rotation, anchors

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
