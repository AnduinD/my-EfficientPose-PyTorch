# Author: Zylo117

import numpy as np
import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, TranslationNet, RotationNet, EfficientNet
#from efficientdet.utils import Anchors

from Transformer import  MultiHeadAttention, create_pad_mask
from Swin_T import WMSA_layer

class EfficientPoseBackbone(nn.Module):
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        #score_threshold = 0.5, # nms后处理用到的
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
        self.anchor_parameters = anchor_parameters
        
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
            *[BiFPN(num_channels = self.bifpn_widths[self.compound_coef],
                    conv_channels = conv_channel_coef[compound_coef],
                    first_time = True if _ == 0 else False, 
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



        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                            #    pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                            #    **kwargs)

        max_size = self.input_sizes[compound_coef]
        from utils.anchors import anchors_for_shape
        self.anchors, self.translation_anchors = anchors_for_shape((max_size,max_size), anchor_params = self.anchor_parameters)
        self.translation_anchors_input = torch.Tensor(np.expand_dims(self.translation_anchors, axis = 0)).cuda() #np.ndarray
        self.anchors_input = torch.Tensor(np.expand_dims(self.anchors, axis = 0)).cuda()  # apply predicted 2D bbox regression to anchors
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # def forward(self, inputs):
    #     input_img, input_cam = inputs
    #     max_size = input_img.shape[-1]

    #     _, p3, p4, p5 = self.backbone_net(input_img)

    #     features = (p3, p4, p5)
    #     features = self.bifpn(features)

    #     regression = self.regressor(features)
    #     classification = self.classifier(features)
    #     translation  = self.translation(features)
    #     rotation = self.rotation([features,input_cam])
        
    #     anchors = self.anchors(input_img, input_img.dtype)

    #     return features, regression, classification, translation, rotation, anchors


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
        
        translation_raw = translation  #torch.tensor
        
        #get anchors and apply predicted translation offsets to translation anchors
        translation_xy_Tz = translation_transform_inv(translation_anchors=self.translation_anchors_input,  #torch.tensor
                                                      deltas= translation_raw)  #torch.tensor
        
        translation_modified = CalculateTxTy(inputs = translation_xy_Tz,  #torch.tensor
                                            fx = input_cam[:, 0],
                                            fy = input_cam[:, 1],
                                            px = input_cam[:, 2],
                                            py = input_cam[:, 3],
                                            tz_scale = input_cam[:, 4],
                                            image_scale = input_cam[:, 5])

        # apply predicted 2D bbox regression to anchors
        bboxes = bbox_transform_inv(boxes = self.anchors_input, deltas = regression[..., :4]) #torch.tensor
        bboxes = ClipBoxes(image = input_img, boxes = bboxes)
        
        #anchors = self.anchors(input_img, input_img.dtype)

        return features, regression, classification, translation_modified, rotation, self.anchors, bboxes
    

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')



def bbox_transform_inv(boxes:torch.Tensor, deltas:torch.Tensor, scale_factors = None):
    """
    Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas of the anchor boxes to the bounding boxes
    Args:
        boxes: Tensor containing the anchor boxes with shape (..., 4)
        deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
        scale_factors: optional scaling factor for the deltas
    Returns:
        Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)

    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = torch.exp(tw) * wa  #w = np.exp(tw) * wa
    h = torch.exp(th) * ha  #h = np.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

def translation_transform_inv(translation_anchors:torch.Tensor, deltas:torch.Tensor, scale_factors = None):
    """ Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors

    Args
        translation_anchors : Tensor of shape (B, N, 3), where B is the batch size, N the number of boxes and 2 values for (x, y) +1 value with the stride.
        deltas: Tensor of shape (B, N, 3). The first 2 deltas (d_x, d_y) are a factor of the stride +1 with Tz.

    Returns
        A tensor of the same shape as translation_anchors, but with deltas applied to each translation_anchors and the last coordinate is the concatenated (untouched) Tz value from deltas.
    """
    stride  = translation_anchors[:, :, -1]
    if scale_factors:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)
    Tz = deltas[:, :, 2]
    pred_translations = torch.stack([x, y, Tz], dim = 2) #x,y 2D Image coordinates and Tz
    return pred_translations

def CalculateTxTy(inputs:torch.Tensor, fx = 572.4114, fy = 573.57043, px = 325.2611, py = 242.04899, tz_scale = 1000.0, image_scale = 1.6666666666666667):
    """ function for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy
        
    # fx = np.expand_dims(fx, axis = -1)
    # fy = np.expand_dims(fy, axis = -1)
    # px = np.expand_dims(px, axis = -1)
    # py = np.expand_dims(py, axis = -1)
    # tz_scale = np.expand_dims(tz_scale, axis = -1)
    # image_scale = np.expand_dims(image_scale, axis = -1)
    fx = torch.unsqueeze(fx, dim = -1)
    fy = torch.unsqueeze(fy, dim = -1)
    px = torch.unsqueeze(px, dim = -1)
    py = torch.unsqueeze(py, dim = -1)
    tz_scale = torch.unsqueeze(tz_scale, dim = -1)
    image_scale = torch.unsqueeze(image_scale, dim = -1)    
    
    x = inputs[:, :, 0] / image_scale
    y = inputs[:, :, 1] / image_scale
    tz = inputs[:, :, 2] * tz_scale
    
    x = x - px
    y = y - py
    
    # tx = np.multiply(x, tz) / fx
    # ty = np.multiply(y, tz) / fy
    # output = np.stack([tx, ty, tz], axis = -1)       
    tx = torch.mul(x, tz) / fx
    ty = torch.mul(y, tz) / fy
    output = torch.stack([tx, ty, tz], dim = -1)    
    return output

def ClipBoxes(image:torch.Tensor, boxes:torch.Tensor):
    """
    Layer that clips 2D bounding boxes so that they are inside the image
    """
    shape = image.shape # [B,C,H,W] 
    height = float(shape[2])
    width = float(shape[3])
    # height = float(shape[1])
    # width = float(shape[2])
    x1 = torch.clip(boxes[:, :, 0], 0, width - 1)
    y1 = torch.clip(boxes[:, :, 1], 0, height - 1)
    x2 = torch.clip(boxes[:, :, 2], 0, width - 1)
    y2 = torch.clip(boxes[:, :, 3], 0, height - 1)

    return torch.stack([x1, y1, x2, y2], dim=2)



class EfficientPoseBackbone_MSA(nn.Module): # MSA: Multihead Self Attention
    '''
    5个attn层放在bifpn和subnet_head之间(参考How Do Vision Transformers Work?)
    '''
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        #score_threshold = 0.5, # nms后处理用到的
                        anchor_parameters = None, # translation后处理用到的
                        num_rotation_parameters = 3,
                        **kwargs):
        super(EfficientPoseBackbone_MSA, self).__init__()
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
        self.anchor_parameters = anchor_parameters

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

       #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                            #    pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                            #    **kwargs)
        max_size = self.input_sizes[compound_coef]
        from utils.anchors import anchors_for_shape
        self.anchors, self.translation_anchors = anchors_for_shape((max_size,max_size), anchor_params = self.anchor_parameters)
        self.translation_anchors_input = torch.Tensor(np.expand_dims(self.translation_anchors, axis = 0)).cuda() #np.ndarray
        self.anchors_input = torch.Tensor(np.expand_dims(self.anchors, axis = 0)).cuda()  # apply predicted 2D bbox regression to anchors


        self.attn_p3 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*64*64
                                       head_size=8,
                                       dropout_rate=0.1)
        
        self.attn_p4 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*32*32
                                        head_size=8,
                                        dropout_rate=0.1)

        self.attn_p5 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*16*16
                                       head_size=8,
                                       dropout_rate=0.1)
        
        self.attn_p6 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*8*8
                                        head_size=8,
                                        dropout_rate=0.1)

        self.attn_p7 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*4*4
                                        head_size=8,
                                        dropout_rate=0.1)
        
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

        features = (self.attn_p3(features[0], features[0], features[0]), 
                    self.attn_p4(features[1], features[1], features[1]), 
                    self.attn_p5(features[2], features[2], features[2]), 
                    self.attn_p6(features[3], features[3], features[3]), 
                    self.attn_p7(features[4], features[4], features[4]))

        regression = self.regressor(features)
        classification = self.classifier(features)
        translation  = self.translation(features)
        rotation = self.rotation([features,input_cam])
        
         # anchors = self.anchors(input_img, input_img.dtype)

        translation_raw = translation  #torch.tensor
        
        #get anchors and apply predicted translation offsets to translation anchors
        translation_xy_Tz = translation_transform_inv(translation_anchors=self.translation_anchors_input,  #torch.tensor
                                                      deltas= translation_raw)  #torch.tensor
        
        translation_modified = CalculateTxTy(inputs = translation_xy_Tz,  #torch.tensor
                                            fx = input_cam[:, 0],
                                            fy = input_cam[:, 1],
                                            px = input_cam[:, 2],
                                            py = input_cam[:, 3],
                                            tz_scale = input_cam[:, 4],
                                            image_scale = input_cam[:, 5])

        # apply predicted 2D bbox regression to anchors
        bboxes = bbox_transform_inv(boxes = self.anchors_input, deltas = regression[..., :4]) #torch.tensor
        bboxes = ClipBoxes(image = input_img, boxes = bboxes)
        
        #anchors = self.anchors(input_img, input_img.dtype)

        return features, regression, classification, translation_modified, rotation, self.anchors, bboxes

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

class EfficientPoseBackbone_MSA_mixed(nn.Module): # MSA: Multihead Self Attention
    '''
    #在efficientnet的骨干里用了MSA替换conv层，参考
    将每个通道的抽头数都改成8（加大头数 参考论文：https://arxiv.org/pdf/2202.06709.pdf）
    '''
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        #score_threshold = 0.5, # nms后处理用到的
                        anchor_parameters = None, # translation后处理用到的
                        num_rotation_parameters = 3,
                        **kwargs):
        super(EfficientPoseBackbone_MSA_mixed, self).__init__()
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
        self.anchor_parameters = anchor_parameters

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

         #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                            #    pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                            #    **kwargs)

        max_size = self.input_sizes[compound_coef]
        from utils.anchors import anchors_for_shape
        self.anchors, self.translation_anchors = anchors_for_shape((max_size,max_size), anchor_params = self.anchor_parameters)
        self.translation_anchors_input = torch.Tensor(np.expand_dims(self.translation_anchors, axis = 0)).cuda() #np.ndarray
        self.anchors_input = torch.Tensor(np.expand_dims(self.anchors, axis = 0)).cuda()  # apply predicted 2D bbox regression to anchors
        
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        self.attn_p3 = MultiHeadAttention(hidden_size=self.input_sizes[self.compound_coef], #B*64*64*64
                                       head_size=8,
                                       dropout_rate=0.1)
        
        self.attn_p4 = MultiHeadAttention(hidden_size=self.input_sizes[self.compound_coef], #B*64*32*32
                                        head_size=8,
                                        dropout_rate=0.1)

        self.attn_p5 = MultiHeadAttention(hidden_size=self.input_sizes[self.compound_coef], #B*64*16*16
                                       head_size=8,
                                       dropout_rate=0.1)
        
        self.attn_p6 = MultiHeadAttention(hidden_size=self.input_sizes[self.compound_coef], #B*64*8*8
                                        head_size=8,
                                        dropout_rate=0.1)

        self.attn_p7 = MultiHeadAttention(hidden_size=self.input_sizes[self.compound_coef], #B*64*4*4
                                        head_size=8,
                                        dropout_rate=0.1)

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

        features = (self.attn_p3(features[0], features[0], features[0]), 
                    self.attn_p4(features[1], features[1], features[1]), 
                    self.attn_p5(features[2], features[2], features[2]), 
                    self.attn_p6(features[3], features[3], features[3]), 
                    self.attn_p7(features[4], features[4], features[4]))

        regression = self.regressor(features)
        classification = self.classifier(features)
        translation  = self.translation(features)
        rotation = self.rotation([features,input_cam])
        
        # anchors = self.anchors(input_img, input_img.dtype)

        translation_raw = translation  #torch.tensor
        
        #get anchors and apply predicted translation offsets to translation anchors
        translation_xy_Tz = translation_transform_inv(translation_anchors=self.translation_anchors_input,  #torch.tensor
                                                      deltas= translation_raw)  #torch.tensor
        
        translation_modified = CalculateTxTy(inputs = translation_xy_Tz,  #torch.tensor
                                            fx = input_cam[:, 0],
                                            fy = input_cam[:, 1],
                                            px = input_cam[:, 2],
                                            py = input_cam[:, 3],
                                            tz_scale = input_cam[:, 4],
                                            image_scale = input_cam[:, 5])

        # apply predicted 2D bbox regression to anchors
        bboxes = bbox_transform_inv(boxes = self.anchors_input, deltas = regression[..., :4]) #torch.tensor
        bboxes = ClipBoxes(image = input_img, boxes = bboxes)
        

        return features, regression, classification, translation_modified, rotation, self.anchors, bboxes


    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')



class EfficientPoseBackbone_WMSA(nn.Module): # WMSA: Windowed Multihead Self Attention
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        #score_threshold = 0.5, # nms后处理用到的
                        anchor_parameters = None, # translation用
                        num_rotation_parameters = 3,
                        **kwargs):
        super(EfficientPoseBackbone_WMSA, self).__init__() # type: ignore
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
        self.anchor_parameters = anchor_parameters

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

        #self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                            #    pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                            #    **kwargs)
        max_size = self.input_sizes[compound_coef]
        from utils.anchors import anchors_for_shape
        self.anchors, self.translation_anchors = anchors_for_shape((max_size,max_size), anchor_params = self.anchor_parameters)
        self.translation_anchors_input = torch.Tensor(np.expand_dims(self.translation_anchors, axis = 0)).cuda() #np.ndarray
        self.anchors_input = torch.Tensor(np.expand_dims(self.anchors, axis = 0)).cuda()  # apply predicted 2D bbox regression to anchors


        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        self.attn_p3 = WMSA_layer(dim=self.bifpn_widths[self.compound_coef], #B*64*64*64
                                  num_heads=8,
                                  attn_drop=0.1)
        
        self.attn_p4 = WMSA_layer(dim=self.bifpn_widths[self.compound_coef], #B*64*32*32
                                  num_heads=8,
                                  attn_drop=0.1)

        self.attn_p5 = WMSA_layer(dim=self.bifpn_widths[self.compound_coef], #B*64*16*16
                                  num_heads=8,
                                  attn_drop=0.1)
        
        self.attn_p6 = WMSA_layer(dim=self.bifpn_widths[self.compound_coef], #B*64*8*8
                                  num_heads=8,
                                  attn_drop=0.1)

        self.attn_p7 = WMSA_layer(dim=self.bifpn_widths[self.compound_coef], #B*64*4*4
                                  num_heads=8,
                                  attn_drop=0.1)        

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

        # 过注意力层 + 短路链接
        features = (self.attn_p3(features[0])+features[0], 
                    self.attn_p4(features[1])+features[1], 
                    self.attn_p5(features[2])+features[2], 
                    self.attn_p6(features[3])+features[3], 
                    self.attn_p7(features[4])+features[4],)
        
        regression = self.regressor(features)
        classification = self.classifier(features)
        translation  = self.translation(features)
        rotation = self.rotation([features,input_cam])
        
        # anchors = self.anchors(input_img, input_img.dtype)

        translation_raw = translation  #torch.tensor
        
        #get anchors and apply predicted translation offsets to translation anchors
        translation_xy_Tz = translation_transform_inv(translation_anchors=self.translation_anchors_input,  #torch.tensor
                                                      deltas= translation_raw)  #torch.tensor
        
        translation_modified = CalculateTxTy(inputs = translation_xy_Tz,  #torch.tensor
                                            fx = input_cam[:, 0],
                                            fy = input_cam[:, 1],
                                            px = input_cam[:, 2],
                                            py = input_cam[:, 3],
                                            tz_scale = input_cam[:, 4],
                                            image_scale = input_cam[:, 5])

        # apply predicted 2D bbox regression to anchors
        bboxes = bbox_transform_inv(boxes = self.anchors_input, deltas = regression[..., :4]) #torch.tensor
        bboxes = ClipBoxes(image = input_img, boxes = bboxes)
        
        #anchors = self.anchors(input_img, input_img.dtype)

        return features, regression, classification, translation_modified, rotation, self.anchors, bboxes

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


class EfficientPoseBackbone_MSA_B(nn.Module): # MSA: Multihead Self Attention
    '''
    5个attn层放在efnet和bifpn之间(参考How Do Vision Transformers Work?)
    '''
    def __init__(self, num_classes=8, 
                        compound_coef=0, 
                        load_weights=False, 
                        num_anchors = 9,
                        freeze_bn = False,
                        #score_threshold = 0.5, # nms后处理用到的
                        anchor_parameters = None, # translation后处理用到的
                        num_rotation_parameters = 3,
                        **kwargs):
        super(EfficientPoseBackbone_MSA, self).__init__()
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
        
        max_size = self.input_sizes[compound_coef]
        from utils.anchors import anchors_for_shape
        self.anchors, self.translation_anchors = anchors_for_shape((max_size,max_size), anchor_params = self.anchor_parameters)
        self.translation_anchors_input = torch.Tensor(np.expand_dims(self.translation_anchors, axis = 0)).cuda() #np.ndarray
        self.anchors_input = torch.Tensor(np.expand_dims(self.anchors, axis = 0)).cuda()  # apply predicted 2D bbox regression to anchors

        self.attn_p3 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*64*64
                                       head_size=8,
                                       dropout_rate=0.1)
        
        self.attn_p4 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*32*32
                                        head_size=8,
                                        dropout_rate=0.1)

        self.attn_p5 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*16*16
                                       head_size=8,
                                       dropout_rate=0.1)
        
        # self.attn_p6 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*8*8
        #                                 head_size=2,
        #                                 dropout_rate=0.1)

        # self.attn_p7 = MultiHeadAttention(hidden_size=self.bifpn_widths[self.compound_coef], #B*64*4*4
        #                                 head_size=1,
        #                                 dropout_rate=0.1)
        
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        input_img, input_cam = inputs
        max_size = input_img.shape[-1]

        _, p3, p4, p5 = self.backbone_net(input_img)

        #       |----|
        # efnet--attn--bifpn
        features = (self.attn_p3(p3,p3,p3)+p3,
                    self.attn_p4(p4,p4,p4)+p4,
                    self.attn_p5(p5,p5,p5)+p5)
        
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        translation  = self.translation(features)
        rotation = self.rotation([features,input_cam])
        
       # anchors = self.anchors(input_img, input_img.dtype)

        translation_raw = translation  #torch.tensor
        
        #get anchors and apply predicted translation offsets to translation anchors
        translation_xy_Tz = translation_transform_inv(translation_anchors=self.translation_anchors_input,  #torch.tensor
                                                      deltas= translation_raw)  #torch.tensor
        
        translation_modified = CalculateTxTy(inputs = translation_xy_Tz,  #torch.tensor
                                            fx = input_cam[:, 0],
                                            fy = input_cam[:, 1],
                                            px = input_cam[:, 2],
                                            py = input_cam[:, 3],
                                            tz_scale = input_cam[:, 4],
                                            image_scale = input_cam[:, 5])

        # apply predicted 2D bbox regression to anchors
        bboxes = bbox_transform_inv(boxes = self.anchors_input, deltas = regression[..., :4]) #torch.tensor
        bboxes = ClipBoxes(image = input_img, boxes = bboxes)

        return features, regression, classification, translation_modified, rotation, self.anchors, bboxes

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')