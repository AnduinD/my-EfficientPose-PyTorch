"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import os,math,time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.backends import cudnn

from backbone import EfficientPoseBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box,postprocess_det
from utils.utils import preprocess_pose, postprocess_pose, get_linemod_camera_matrix, get_linemod_3d_bboxes
from utils.visualization import draw_detections

compound_coef = 0  # 耦合因子φ
force_input_size = None  # set None to use default size
#img_path = 'test/img.png'

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2**0, 2**(1.0/3.0), 2**(2.0/3.0)]

score_threshold = 0.5
iou_threshold = 0.2

use_cuda = True #False #
use_float16 = False
cudnn.fastest = True  # type: ignore
cudnn.benchmark = True # type: ignore

class ModelWithFilterDet(nn.Module):
    def __init__(self, model):
        super(ModelWithFilterDet, self).__init__()
        self.model = model
        self.filter_det = FilterDetections(            
                                num_rotation_parameters = 3,
                                num_translation_parameters = 3,
                                nms = True,
                                class_specific_filter = True,
                                nms_threshold = 0.5, 
                                score_threshold = 0.01,
                                max_detections = 100)

    def forward(self, inputs):
        features, regression, classification, translation, rotation, anchors = self.model(inputs)
        regression, classification, translation, rotation = self.filter_det(regression, classification, translation, rotation)
        return features, regression, classification, translation, rotation, anchors

def main():
    #input parameter
    path_to_images = "../datasets/Linemod_preprocessed/data/08/rgb/"
    image_extension = ".png"
    path_to_weights = f'weights/trained/efficientpose-d{compound_coef}_linemod_obj8_one_last_train.pth'
    batch_size = 2
    save_path = "./predictions" #where to save the images or None if the images should be displayed and not saved
    #class_to_name = {0: "ape", 1: "can", 2: "cat", 3: "driller", 4: "duck", 5: "eggbox", 6: "glue", 7: "holepuncher"} #Occlusion
    class_to_name = {0: "driller"} #Linemod use a single class with a name of the Linemod objects
    
    translation_scale_norm = 1000.0
    draw_bbox_2d = False
    draw_name = False
    #for the linemod and occlusion trained models take this camera matrix and these 3d models. in case you trained a model on a custom dataset you need to take the camera matrix and 3d cuboids from your custom dataset.
    camera_matrix = get_linemod_camera_matrix()
    name_to_3d_bboxes = get_linemod_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    
    num_classes = len(class_to_name)
    
    if not os.path.exists(path_to_images):
        print("Error: the given path to the images {} does not exist!".format(path_to_images))
        return
    
    image_list = [filename for filename in os.listdir(path_to_images) if image_extension in filename]
    print("\nInfo: found {} image files".format(len(image_list)))   
    
    #build model and load weights
    model = EfficientPoseBackbone(
        compound_coef=compound_coef, 
        num_classes=num_classes,
        ratios=anchor_ratios, 
        scales=anchor_scales)
    
    #print(model)

    temp_weight = torch.load(path_to_weights, map_location='cpu')
    # del temp_weight['classifier.header.pointwise_conv.conv.weight']
    # del temp_weight['classifier.header.pointwise_conv.conv.bias']
    model.load_state_dict(temp_weight, strict = False) # 类别数变了 删掉这部分权重再load
 
    model = ModelWithFilterDet(model)

    model.requires_grad_(False)
    model.eval()


    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()


    #inferencing
    with torch.no_grad():
        input_list = []
        input_batch_list = [[],[]]
        scale_batch_list = []
        img_org_path_list = []
        
        print("load images...")
        for image_filename in tqdm(image_list):
            #load image
            image_path = os.path.join(path_to_images, image_filename)
            image = cv2.imread(image_path)
            #original_image = image.copy()
            
            #preprocessing
            image_size = model.input_sizes[compound_coef] # type: ignore
            input_list, scale = preprocess_pose(image, image_size, camera_matrix, translation_scale_norm)

            input_batch_list[0].append(input_list[0]) # img_batch_list
            input_batch_list[1].append(input_list[1]) # cam_batch_list
            img_org_path_list.append(image_path)
            scale_batch_list.append(scale)  # 注：目前这个参数在后处理中没有用到  怀疑用不到它 在网络正向里面有在生成anchor，所以我怀疑不需要再在后处理做缩放？
        
        #predict
        for idx_i in tqdm(range(0,len(image_list),batch_size)) :
            if use_cuda:
                input_img_batch = torch.cat([torch.from_numpy(img).cuda() for img in 
                                             input_batch_list[0][idx_i:idx_i+batch_size]], 0)
                input_cam_batch = torch.cat([torch.from_numpy(cam).cuda() for cam in 
                                             input_batch_list[1][idx_i:idx_i+batch_size]], 0)
            else:
                input_img_batch = torch.cat([torch.from_numpy(img) for img in 
                                             input_batch_list[0][idx_i:idx_i+batch_size]], 0)
                input_cam_batch = torch.cat([torch.from_numpy(cam) for cam in 
                                             input_batch_list[1][idx_i:idx_i+batch_size]], 0)

            input_img_batch = input_img_batch.to(torch.float32 if not use_float16 
                                                 else torch.float16).permute(0,3,1,2)
            
            # boxes, scores, labels, rotations, translations, anchors = model((input_imgs,input_cams)) 
            features, regression, classification, translation, rotation, anchors = model((input_img_batch,input_cam_batch))

            #print("after inference")
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out_dict = postprocess_pose(input_img_batch,
                                       anchors, 
                                       regression, 
                                       classification,
                                       translation,
                                       rotation,
                                       regressBoxes, 
                                       clipBoxes,
                                       scale_batch_list,
                                       score_threshold, 
                                       iou_threshold)
                
            for idx_j in range(idx_i, idx_i+batch_size) :
                org_image = cv2.imread(img_org_path_list[idx_j])
                draw_detections(org_image,
                                out_dict[idx_j-idx_i]["rois"],
                                out_dict[idx_j-idx_i]["scores"],
                                out_dict[idx_j-idx_i]["class_ids"],
                                out_dict[idx_j-idx_i]["rotations"],
                                out_dict[idx_j-idx_i]["translations"],
                                class_to_bbox_3D = class_to_3d_bboxes,
                                camera_matrix = camera_matrix,
                                label_to_name = class_to_name,
                                draw_bbox_2d = draw_bbox_2d,
                                draw_name = draw_name)
                
                if save_path is None:
                    #display image with predictions
                    cv2.imshow('image with predictions', org_image)
                    cv2.waitKey(0)
                else:
                    #only save images to the given path
                    os.makedirs(save_path, exist_ok = True)
                    cv2.imwrite(os.path.join(save_path, image_list[idx_j].replace(image_extension, "_predicted" + image_extension)), org_image)

            
            



if __name__ == '__main__':
    main()
