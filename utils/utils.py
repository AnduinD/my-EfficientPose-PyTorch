# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union

import cv2
import numpy as np
import torch
import webcolors
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torchvision.ops.boxes import batched_nms

from .sync_batchnorm import SynchronizedBatchNorm2d


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]  #type:ignore
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h: # 约束长宽比的resize
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h: # 如果要resize
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 前面向下取整时留下的padding位置
    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess_det(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]  # 读图片
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]  #做归一化均值方差 #type:ignore
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs] 
                                            # 没太懂 似乎是resize+pad到512的处理
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]   # 记录whc的图片矩阵
    framed_metas = [img_meta[1:] for img_meta in imgs_meta] # 记录图片元信息的tuple表
    return ori_imgs, framed_imgs, framed_metas

def preprocess_det_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def postprocess_det(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out

# def postprocess_pose(boxes, scores, labels, rotations, translations, scale, score_threshold):
#     """
#     Filter out detections with low confidence scores and rescale the outputs of EfficientPose
#     Args:
#         boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
#         scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
#         labels: numpy array [batch_size = 1, max_detections] containing class label
#         rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
#         translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
#         scale: The scale factor of the resized input image and the original image
#         score_threshold: Minimum score threshold at which a prediction is not filtered out
#     Returns:
#         boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
#         scores: numpy array [num_valid_detections] containing the confidence scores
#         labels: numpy array [num_valid_detections] containing class label
#         rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
#         translations: numpy array [num_valid_detections, 3] containing the translation vectors

#     """
#     boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
#     # correct boxes for image scale
#     boxes /= scale
#     #rescale rotations
#     rotations *= math.pi
#     #filter out detections with low scores
#     indices = np.where(scores[:] > score_threshold)
#     # select detections
#     scores = scores[indices]
#     boxes = boxes[indices]
#     rotations = rotations[indices]
#     translations = translations[indices]
#     labels = labels[indices]
    
#     return boxes, scores, labels, rotations, translations


def postprocess_pose(input_imgs, anchors, 
                    regression, classification, translation, rotation, 
                    regressBoxes, clipBoxes, scale_batch_list,
                    score_threshold, iou_threshold):
    # regression /= scale_batch_list[0]
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, input_imgs)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > score_threshold)[:, :, 0]

    rotation *= math.pi

    out = []
    for i in range(input_imgs.shape[0]):  # 对batch做遍历
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'translations': np.array(()),
                'rotations': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        translation_per = translation[i, scores_over_thresh[i, :], ...]
        rotation_per = rotation[i, scores_over_thresh[i, :], ...]

        anchors_nms_idx = batched_nms(boxes = transformed_anchors_per,
                                      scores =  scores_per[:, 0], 
                                      idxs = classes_, 
                                      iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]
            translations_ = translation_per[anchors_nms_idx,:] #shape:[anchors_nms_idx,3]
            rotations_ = rotation_per[anchors_nms_idx,:] #shape:[anchors_nms_idx,3]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
                'translations': translations_.cpu().numpy(),
                'rotations': rotations_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
                'translations': np.array(()),
                'rotations': np.array(()),
            })

    return out #返回一个大list，list里的元素是一张图里所有对象信息的dict
    #return out[0] #返回一一张图里所有对象信息的dict


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight #type:ignore
                target_attr.bias = bias #type:ignore

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel): #type: ignore
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (torch.Tensor, float) -> torch.Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


color_list = standard_to_bgr(STANDARD_COLORS)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


#############################EfficientPose_tf1#########################################
def get_linemod_camera_matrix():
    """
    Returns:
        The Linemod and Occlusion 3x3 camera matrix
    """
    return np.array([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]], dtype = np.float32)

def get_linemod_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values
    """
    name_to_model_info = {"ape":            {"diameter": 102.09865663, "min_x": -37.93430000, "min_y": -38.79960000, "min_z": -45.88450000, "size_x": 75.86860000, "size_y": 77.59920000, "size_z": 91.76900000},
                            "benchvise":    {"diameter": 247.50624233, "min_x": -107.83500000, "min_y": -60.92790000, "min_z": -109.70500000, "size_x": 215.67000000, "size_y": 121.85570000, "size_z": 219.41000000},
                            "cam":          {"diameter": 172.49224865, "min_x": -68.32970000, "min_y": -71.51510000, "min_z": -50.24850000, "size_x": 136.65940000, "size_y": 143.03020000, "size_z": 100.49700000},
                            "can":          {"diameter": 201.40358597, "min_x": -50.39580000, "min_y": -90.89790000, "min_z": -96.86700000, "size_x": 100.79160000, "size_y": 181.79580000, "size_z": 193.73400000},
                            "cat":          {"diameter": 154.54551808, "min_x": -33.50540000, "min_y": -63.81650000, "min_z": -58.72830000, "size_x": 67.01070000, "size_y": 127.63300000, "size_z": 117.45660000},
                            "driller":      {"diameter": 261.47178102, "min_x": -114.73800000, "min_y": -37.73570000, "min_z": -104.00100000, "size_x": 229.47600000, "size_y": 75.47140000, "size_z": 208.00200000},
                            "duck":         {"diameter": 108.99920102, "min_x": -52.21460000, "min_y": -38.70380000, "min_z": -42.84850000, "size_x": 104.42920000, "size_y": 77.40760000, "size_z": 85.69700000},
                            "eggbox":       {"diameter": 164.62758848, "min_x": -75.09230000, "min_y": -53.53750000, "min_z": -34.62070000, "size_x": 150.18460000, "size_y": 107.07500000, "size_z": 69.24140000},
                            "glue":         {"diameter": 175.88933422, "min_x": -18.36050000, "min_y": -38.93300000, "min_z": -86.40790000, "size_x": 36.72110000, "size_y": 77.86600000, "size_z": 172.81580000},
                            "holepuncher":  {"diameter": 145.54287471, "min_x": -50.44390000, "min_y": -54.24850000, "min_z": -45.40000000, "size_x": 100.88780000, "size_y": 108.49700000, "size_z": 90.80000000},
                            "iron":         {"diameter": 278.07811733, "min_x": -129.11300000, "min_y": -59.24100000, "min_z": -70.56620000, "size_x": 258.22600000, "size_y": 118.48210000, "size_z": 141.13240000},
                            "lamp":         {"diameter": 282.60129399, "min_x": -101.57300000, "min_y": -58.87630000, "min_z": -106.55800000, "size_x": 203.14600000, "size_y": 117.75250000, "size_z": 213.11600000},
                            "phone":        {"diameter": 212.35825148, "min_x": -46.95910000, "min_y": -73.71670000, "min_z": -92.37370000, "size_x": 93.91810000, "size_y": 147.43340000, "size_z": 184.74740000}}
        
    name_to_3d_bboxes = {name: convert_bbox_3d(model_info) for name, model_info in name_to_model_info.items()}
    
    return name_to_3d_bboxes

def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.
    """
    #get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]
    
    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox

def preprocess_pose(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for EfficientPose
    Args:
        image: The image to predict
        image_size: Input resolution for EfficientPose
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
    Returns:
        input_list: List containing the preprocessed inputs for EfficientPose
        scale: The scale factor of the resized input image and the original image
    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale

def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]
    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector

# def postprocess_pose(boxes, scores, labels, rotations, translations, scale, score_threshold):
#     """
#     Filter out detections with low confidence scores and rescale the outputs of EfficientPose
#     Args:
#         boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
#         scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
#         labels: numpy array [batch_size = 1, max_detections] containing class label
#         rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
#         translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
#         scale: The scale factor of the resized input image and the original image
#         score_threshold: Minimum score threshold at which a prediction is not filtered out
#     Returns:
#         boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
#         scores: numpy array [num_valid_detections] containing the confidence scores
#         labels: numpy array [num_valid_detections] containing class label
#         rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
#         translations: numpy array [num_valid_detections, 3] containing the translation vectors
#     """
#     boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
#     # correct boxes for image scale
#     boxes /= scale
#     #rescale rotations
#     rotations *= math.pi
#     #filter out detections with low scores
#     indices = np.where(scores[:] > score_threshold)
#     # select detections
#     scores = scores[indices]
#     boxes = boxes[indices]
#     rotations = rotations[indices]
#     translations = translations[indices]
#     labels = labels[indices]
    
#     return boxes, scores, labels, rotations, translations

def preprocess_image(image, image_size): #efpose_tf1的
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant') # type: ignore
    return image, scale


def rotate_image(image):
    rotate_degree = np.random.uniform(low=-45, high=45)
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))

    return image


def reorder_vertexes(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    Args:
        vertexes: np.array (4, 2), should be in clockwise

    Returns:

    """
    assert vertexes.shape == (4, 2)
    xmin, ymin = np.min(vertexes, axis=0)
    xmax, ymax = np.max(vertexes, axis=0)

    # determine the first point with the smallest y,
    # if two vertexes has same y, choose that with smaller x,
    ordered_idxes = np.argsort(vertexes, axis=0)
    ymin1_idx = ordered_idxes[0, 1]
    ymin2_idx = ordered_idxes[1, 1]
    if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
        if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
            first_vertex_idx = ymin1_idx
        else:
            first_vertex_idx = ymin2_idx
    else:
        first_vertex_idx = ymin1_idx
    ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
    ordered_vertexes = vertexes[ordered_idxes]
    # drag the point to the corresponding edge
    ordered_vertexes[0, 1] = ymin
    ordered_vertexes[1, 0] = xmax
    ordered_vertexes[2, 1] = ymax
    ordered_vertexes[3, 0] = xmin
    return ordered_vertexes


def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes





def gather_torch(params, indices, axis=None):
  dim = axis
  new_size = params.size()[:dim] + indices.size() + params.size()[dim+1:] #type:ignore
  out = params.index_select(index=indices.view(-1), dim=dim)
  out.view(new_size)   
  return out


def gather_nd_batch(params, indices, batch_dim=1):
    """ A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = gather_nd_torch(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    gather_dims.insert(0, batch_enumeration)
    gathered = params[gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered

def gather_nd_simple(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)