# my test with EfficientPose in Pytorch

## Demo

    # install requirements
    pip install pycocotools numpy opencv-python tqdm tensorboard pyyaml webcolors
    pip install torch==1.12.1+cu113
    pip install torchvision==0.13.0+cu113
     
    # run the simple inference script
    python inference.py

## Training

### Train a custom dataset with pretrained weights (Highly Recommended)

    python ./train_pose_WMSA.py --weights ./weights/trained_WMSA/obj_8/efficientpose-d0_linemod_obj8_one_best_train.pth  --lr  1e-3  --batch_size 2

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    python evaluate_WMSA.py --object_id 8  --weights  ./weights/trained_WMSA/obj_8/efficientpose-d0_linemod_obj8_one_best_train.pth





## References

Appreciate the great work from the following repositories:

- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)
- [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
- https://github.com/claire-s11
- https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

