
import argparse, time, os, sys, traceback
import datetime
from tqdm.autonotebook import tqdm
import numpy as np
import torch
import torch.nn as nn
#from tensorflow.keras.optimizers import Adam
from torch.utils.tensorboard import SummaryWriter #type:ignore

from backbone import EfficientPoseBackbone, EfficientPoseBackbone_MSA_mixed 
from torch.nn import SmoothL1Loss
from efficientdet.loss import FocalLoss
from loss_6DoF import  transformation_loss #smooth_l1,
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import  CustomDataParallel, replace_w_sync_bn

def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'Simple EfficientPose training script.')
    #subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    #subparsers.required = True

    parser.add_argument('--dataset_type', 
                        default = 'linemod', 
                        help = 'Arguments for specific dataset types.')    
    # linemod_parser = subparsers.add_parser('linemod')
    # linemod_parser.add_argument('linemod_path', 
    #                             default='../datasets/Linemod_preprocessed', 
    #                             help = 'Path to dataset dir (ie. /Datasets/Linemod_preprocessed).')
    # linemod_parser.add_argument('--object-id', 
    #                             type = int, default = 8, 
    #                             help = 'ID of the Linemod Object to train on')
    parser.add_argument('--linemod_path', 
                        default='../datasets/Linemod_preprocessed', 
                        help = 'Path to dataset dir (ie. /Datasets/Linemod_preprocessed).')
    parser.add_argument('--object-id', 
                        type = int, default = 8, 
                        help = 'ID of the Linemod Object to train on')    


    # occlusion_parser = subparsers.add_parser('occlusion')
    # occlusion_parser.add_argument('occlusion_path',  
    #                               default='../datasets/Linemod_preprocessed',
    #                               help = 'Path to dataset dir (ie. /Datasets/Linemod_preprocessed).')

    parser.add_argument('--weights',
                        #default= 'imagenet',
                        default='./weights/trained_MSA_Mixed/efficientpose-d0_linemod_obj8_one_best_train.pth', 
                        help = 'File containing weights to init the model parameter')
    parser.add_argument('--save_path', 
                        help = 'path where to save the predicted validation images after each epoch', 
                        default = './weights/trained_MSA_Mixed')
    parser.add_argument('--es_patience',
                        help='patience for early stopping',
                        default=25, type=int)
    parser.add_argument('--save_interval',
                        help='interval for saving model',
                        default=1000, type=int)
    parser.add_argument('--val_interval',
                        help='interval for validation',
                        default = 10, type=int)
    
    parser.add_argument('--freeze_backbone', 
                        help = 'Freeze training of backbone layers.', 
                        action = 'store_true')
    parser.add_argument('--no_freeze_bn', 
                        help = 'Do not freeze training of BatchNormalization layers.',
                        action = 'store_true')

    parser.add_argument('--batch_size',
                        help = 'Size of the batches.',
                        default = 2, type = int)
    parser.add_argument('--lr', 
                        help = 'Learning rate',
                        default = 1e-3, type = float)
    parser.add_argument('--no_color_augmentation', 
                        help = 'Do not use colorspace augmentation', 
                        action = 'store_true', default = False)
    parser.add_argument('--no_6dof_augmentation', 
                        help = 'Do not use 6DoF augmentation', 
                        action = 'store_true', default = False)
    parser.add_argument('--rotation_representation', 
                        default = 'axis_angle', 
                        help = 'Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"')    
    parser.add_argument('--phi', 
                        help = 'Hyper parameter phi', 
                        default = 0, type = int, 
                        choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', 
                        help = 'Id of the GPU to use (as reported by nvidia-smi).',
                        default = [0,])
    parser.add_argument('--num_workers', 
                        help = 'Number of workers used in dataloading',
                        default = 0, type = int)
    parser.add_argument('--epochs', 
                        help = 'Number of epochs to train.', 
                        default = 500, type = int)
    # parser.add_argument('--steps', 
    #                     help = 'Number of steps per epoch.', 
    #                     type = int, default = int(179 * 10))

    # parser.add_argument('--snapshot_path', 
    #                     help = 'Path to store snapshots of models during training', 
    #                     default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--log_path', 
                        help = 'Log directory for Tensorboard output', 
                        default = os.path.join("logs", date_and_time))
    # parser.add_argument('--no_snapshots', 
    #                     help = 'Disable saving snapshots.', 
    #                     dest = 'snapshots', 
    #                     action = 'store_false')
    # parser.add_argument('--no-evaluation', 
    #                     help = 'Disable per epoch evaluation.', 
    #                     dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute_val_loss', 
                        help = 'Compute validation loss during training', 
                        dest = 'compute_val_loss', action = 'store_true',
                        default=True)
    parser.add_argument('--score_threshold', 
                        help = 'score threshold for non max suppresion', 
                        type = float, default = 0.5)

    # # Fit generator arguments
    # parser.add_argument('--multiprocessing', 
    #                     help = 'Use multiprocessing in fit_generator.', 
    #                     action = 'store_true')
    # parser.add_argument('--workers', 
    #                     help = 'Number of generator workers.', 
    #                     type = int, default = 4)
    # parser.add_argument('--max-queue-size', 
    #                     help = 'Queue length for multiprocessing workers in fit_generator.', 
    #                     type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)



class ModelWithLoss(nn.Module):
    def __init__(self, model, param_generator, debug=False, num_gpus=0):
        super().__init__()
        self.criterion_focal = FocalLoss(alpha = 0.25, gamma = 2.0)
        self.criterion_l1 = SmoothL1Loss(beta=3.0)
        self.criterion_transformation = transformation_loss(
            model_3d_points_np= param_generator.get_all_3d_model_points_array_for_loss(),
            num_rotation_parameter = param_generator.get_num_rotation_parameters(),
            num_gpus = num_gpus)
        self.model = model
        self.debug = debug

    def forward(self, inputs, annotations, obj_list=None):
        features, regression, classification, translation, rotation, anchors, _ = self.model(inputs)
        
        from loss_6DoF import gather_nd_simple
        if self.debug:
            cls_loss = self.criterion_focal(classification,
                                            annotations[0],
                                            imgs=inputs[0], 
                                            obj_list=obj_list)
            smooth_l1_loss = self.criterion_l1(regression,
                                               annotations[1])
            transformation_loss = self.criterion_transformation(rotation, 
                                                                translation, 
                                                                annotations[2])

        else:
            anchor_states = annotations[0][:,:,-1]
            indices = anchor_states.ne(-1).nonzero() #找出那些有效的锚框（anno的最后一个值为-1都表示忽略）
            ##################
            cls_loss = self.criterion_focal(gather_nd_simple(classification, indices),
                                            gather_nd_simple(annotations[0][:,:,:-1], indices))
            smooth_l1_loss = self.criterion_l1(gather_nd_simple(regression, indices), 
                                               gather_nd_simple(annotations[1][:,:,:-1], indices))
            transformation_loss = self.criterion_transformation(rotation, 
                                                                translation, 
                                                                annotations[2])
        
        return 1.0*smooth_l1_loss+1.0*cls_loss+0.02*transformation_loss, \
                cls_loss, smooth_l1_loss, transformation_loss



def main(args = None):
    """
    Train an EfficientPose model.
    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    
    args = parse_args(args)
    args.num_gpus = len(args.gpu) if args.gpu else 0    
    args.save_path = os.path.join(args.save_path, f'obj_{args.object_id}')

    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")
    
    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors
    
    if args.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("\nBuilding the Model...")

    #build model and load weights
    model = EfficientPoseBackbone_MSA_mixed(compound_coef=args.phi, 
                                            num_classes=num_classes,
                                            num_anchors=num_anchors,
                                            freeze_bn=not args.no_freeze_bn,
                                            #score_threshold = args.score_threshold,
                                            num_rotation_parameters = num_rotation_parameters)

    print("Done!")
    # load pretrained weights
    if args.weights:
        if args.weights == 'imagenet':
            print('Loading model, this may take a second...')
            model_name = 'efficientdet-d{}'.format(args.phi)
            file_name = '{}.pth'.format(model_name)
            weights_path = f'weights/pretrained_efficientdet/{file_name}'
            temp_weight = torch.load(weights_path, map_location='cpu')
            del temp_weight['classifier.header.pointwise_conv.conv.weight']
            del temp_weight['classifier.header.pointwise_conv.conv.bias']
            model.load_state_dict(temp_weight, strict = False) # 类别数变了 删掉这部分权重再load
            # temp_weight = torch.load(args.weights, map_location='cpu')
            # model.load_state_dict(temp_weight, strict = False)
            print("\nDone!")
        else:
            model.load_state_dict(torch.load(args.weights), strict=False)
            print('Loading model, this may take a second...')
            temp_weight = torch.load(args.weights, map_location='cpu')
            model.load_state_dict(temp_weight, strict = False) 
            print("\nDone!")


    # freeze backbone layers
    if args.freeze_backbone:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False
        model.apply(freeze_backbone)
        print('[Info] freezed backbone')
        # 227, 329, 329, 374, 464, 566, 656
        # for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
        #     model.layers[i].trainable = False

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    writer = SummaryWriter(args.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
    
    if args.num_gpus > 1 and args.batch_size // args.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    model = ModelWithLoss(model, param_generator=train_generator ,debug=False, num_gpus = args.num_gpus)

    if args.num_gpus > 0:
        model = model.cuda()
        if args.num_gpus > 1:
            model = CustomDataParallel(model, args.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (0.9, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas = (0.9, 0.999), amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor = 0.5, verbose=True)

    epoch = 0
    step = 0
    best_loss = 1e5
    best_epoch = 0
    val_interval = args.val_interval
    model.train()

    num_iter_per_epoch = len(train_generator)

    try:
        for epoch in range(args.epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(train_generator) # type: ignore
            for iter, data in enumerate(progress_bar):
                '''
                #data[0][0]:batch of imgs (batchsize,W,H,C), data[0][1]:batch of cam param K (batchsize,6)
                #data[1][0]:batch of labels(batchsize,num_anchors,num_classes+1) , data[1][1]:batch of 2d regresion (batchsize,num_anchors,4+1)
                #data[1][2]:batch of 3d transformation (batchsize,num_anchors,num_rotation_parameters + num_translation_parameters + 1)
                '''
                
                # continue
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                if iter >= num_iter_per_epoch: 
                    break # 一个epoch结束 因为这个generator会循环生成 所以得手动退出
                try:
                    imgs = torch.Tensor(data[0][0]).permute(0,3,1,2) # BWHC -> BCHW
                    cams_K = torch.Tensor(data[0][1])
                    annot = [torch.Tensor(sub_anno) for sub_anno in data[1]] # 3组标签：类别、检测框、3d变换

                    if args.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        cams_K = cams_K.cuda()
                        annot = [sub_anno.cuda() for sub_anno in annot]

                    optimizer.zero_grad()
                    loss, cls_loss, reg_loss, transformation_loss = model((imgs,cams_K), annot) #obj_list=args.obj_list
                    
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. 3D loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, args.epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), transformation_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    writer.add_scalars('Transformation_loss', {'train': transformation_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % args.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_{step}.pth',args.save_path)
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            
            if epoch % val_interval == 0:
                model.eval()
                len_validation_generator = len(validation_generator) #type:ignore
                loss_regression_ls = []
                loss_classification_ls = []
                loss_transformation_ls = []
                print(f"validating...")
                for iter, data in enumerate(tqdm(validation_generator)): # type: ignore
                    if iter >= len_validation_generator: 
                        break # 一个epoch结束 因为这个generator会循环生成 所以得手动退出
                    with torch.no_grad():
                        imgs = torch.Tensor(data[0][0]).permute(0,3,1,2) # BWHC -> BCHW
                        cams_K = torch.Tensor(data[0][1])
                        annot = [torch.Tensor(sub_anno) for sub_anno in data[1]] # 3组标签：类别、检测框、3d变换

                        if args.num_gpus == 1:
                            # if only one gpu, just send it to cuda:0
                            # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                            imgs = imgs.cuda()
                            cams_K = cams_K.cuda()
                            annot = [sub_anno.cuda() for sub_anno in annot]

                        optimizer.zero_grad()
                        loss, cls_loss, reg_loss, transformation_loss = model((imgs,cams_K), annot) #obj_list=args.obj_list
                        
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())
                        loss_transformation_ls.append(transformation_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                transformation_loss = np.mean(loss_transformation_ls)
                loss = 1.0*cls_loss + 1.0*reg_loss + 0.02*transformation_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Transformation loss: {:.5f}. Total loss: {:1.5f}'.format(
                        epoch, args.epochs, cls_loss, reg_loss, transformation_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                writer.add_scalars('Transformation_loss', {'val': transformation_loss}, step)

                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_{step}.pth',args.save_path)
                    save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_one_best_train.pth',args.save_path)

                model.train()

                # Early stopping
                if epoch - best_epoch > args.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_{step}.pth',args.save_path)
                    save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_one_last_train.pth',args.save_path)
                    break

    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientpose-d{args.phi}_{args.dataset_type}_obj{args.object_id}_{step}.pth', args.save_path)
        writer.close()
    writer.close()


def save_checkpoint(model, name, save_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(save_path, name)) #type: ignore
    else:
        torch.save(model.model.state_dict(), os.path.join(save_path, name))

def create_generators(args):
    """
    Create generators for training and validation.
    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
        #'shuffle':True,
        #'drop_last':True,
        #'num_workers': args.num_workers,
    }

    if args.dataset_type == 'linemod':
        from generators.linemod import LineModGenerator
        train_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'occlusion':
        from generators.occlusion import OcclusionGenerator
        train_generator = OcclusionGenerator(
            args.linemod_path,#args.occlusion_path,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = OcclusionGenerator(
            args.linemod_path,#args.occlusion_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
