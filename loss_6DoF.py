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
"""

# from tensorflow import keras
# import tensorflow as tf
import math
import numpy as np
import torch
import torch.nn as nn
from utils.utils import gather_nd_simple, gather_torch # gather_nd_batch
# from torch.nn import SmoothL1Loss
# from efficientdet.loss import FocalLoss

# def focal(alpha=0.25, gamma=1.5):
#     """
#     Create a functor for computing the focal loss.

#     Args:
#         alpha: Scale the focal weight with alpha.
#         gamma: Take the power of the focal weight with gamma.

#     Returns:
#         A functor that computes the focal loss using the alpha and gamma.
#     """

#     def _focal(y_true, y_pred):
#         """
#         Compute the focal loss given the target tensor and the predicted tensor.

#         As defined in https://arxiv.org/abs/1708.02002

#         Args:
#             y_true: Tensor of target data from the generator with shape (B, N, num_classes).
#             y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

#         Returns:
#             The focal loss of y_pred w.r.t. y_true.
#         """
#         labels = y_true[:, :, :-1]
#         # -1 for ignore, 0 for background, 1 for object
#         anchor_state = y_true[:, :, -1]
#         classification = y_pred

#         # filter out "ignore" anchors
#         indices = tf.where(keras.backend.not_equal(anchor_state, -1))
#         labels = gather_nd_torch(labels, indices)
#         classification = gather_nd_torch(classification, indices)

#         # compute the focal loss
#         alpha_factor = keras.backend.ones_like(labels) * alpha
#         alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
#         # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
#         focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
#         focal_weight = alpha_factor * focal_weight ** gamma
#         cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

#         # compute the normalizer: the number of positive anchors
#         normalizer = tf.where(keras.backend.equal(anchor_state, 1))
#         normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
#         normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

#         return keras.backend.sum(cls_loss) / normalizer

#     return _focal


# def smooth_l1(sigma=3.0):
#     """
#     Create a smooth L1 loss functor.
#     Args:
#         sigma: This argument defines the point where the loss changes from L2 to L1.
#     Returns:
#         A functor for computing the smooth L1 loss given target data and predicted data.
#     """
#     sigma_squared = sigma ** 2

#     def _smooth_l1(y_true, y_pred):
#         """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
#         Args:
#             y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
#             y_pred: Tensor from the network of shape (B, N, 4).
#         Returns:
#             The smooth L1 loss of y_pred w.r.t. y_true.
#         """
#         # separate target and state
#         regression = y_pred
#         regression_target = y_true[:, :, :-1]
#         anchor_state = y_true[:, :, -1]

#         # filter out "ignore" anchors
#         indices = torch.where(torch.equal(anchor_state, 1))
#         regression = gather_nd_torch(regression, indices)
#         regression_target = gather_nd_torch(regression_target, indices)

#         # compute smooth L1 loss
#         # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
#         #        |x| - 0.5 / sigma / sigma    otherwise
#         regression_diff = regression - regression_target
#         regression_diff = keras.backend.abs(regression_diff)
#         regression_loss = tf.where(
#             keras.backend.less(regression_diff, 1.0 / sigma_squared),
#             0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
#             regression_diff - 0.5 / sigma_squared
#         )

#         # compute the normalizer: the number of positive anchors
#         normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
#         normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
#         return keras.backend.sum(regression_loss) / normalizer

#     return _smooth_l1

class transformation_loss(nn.Module):
    """
    Create a transformation loss functor as described in https://arxiv.org/abs/2011.04307
    Args:
        model_3d_points_np: numpy array containing the 3D model points of all classes for calculating the transformed point distances.
                            The shape is (num_classes, num_points, 3)
        num_rotation_parameter: The number of rotation parameters, usually 3 for axis angle representation
    Returns:
        A functor for computing the transformation loss given target data and predicted data.
    """
    def __init__(self, model_3d_points_np, num_rotation_parameter, num_gpus = 0):
        super(transformation_loss, self).__init__()
  
        self.model_3d_points = torch.Tensor(model_3d_points_np).cuda() if num_gpus > 0 else torch.Tensor(model_3d_points_np)
        self.num_rotation_parameter = num_rotation_parameter
        self.num_points = self.model_3d_points.shape[1]    
    
    def forward(self, regression_rotation,regression_translation, anno):
        """ Compute the transformation loss of y_pred w.r.t. y_true using the model_3d_points tensor.
        Args:
            y_true: Tensor from the generator of shape (B, N, num_rotation_parameter + num_translation_parameter + is_symmetric_flag + class_label + anchor_state).
                    num_rotation_parameter is 3 for axis angle representation and num_translation parameter is also 3
                    is_symmetric_flag is a Boolean indicating if the GT object is symmetric or not, used to calculate the correct loss
                    class_label is the class of the GT object, used to take the correct 3D model points from the model_3d_points tensor for the transformation
                    The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, num_rotation_parameter + num_translation_parameter).
        Returns:
            The transformation loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        # regression_rotation = y_pred[:, :, :self.num_rotation_parameter]
        # regression_translation = y_pred[:, :, self.num_rotation_parameter:]
        regression_target_rotation = anno[:, :, :self.num_rotation_parameter]
        regression_target_translation = anno[:, :, self.num_rotation_parameter:-3]
        is_symmetric = anno[:, :, -3]
        class_indices = anno[:, :, -2]
        anchor_state  = torch.round(anno[:, :, -1]).type(torch.int32)
    
        # filter out "ignore" anchors
        indices = torch.nonzero(torch.eq(anchor_state, 1))
        regression_rotation = gather_nd_simple(regression_rotation,indices) * math.pi
        regression_translation = gather_nd_simple(regression_translation, indices)
        
        regression_target_rotation = gather_nd_simple(regression_target_rotation, indices) * math.pi
        regression_target_translation = gather_nd_simple(regression_target_translation, indices)
        is_symmetric = gather_nd_simple(is_symmetric, indices)
        is_symmetric =torch.round(is_symmetric).type(torch.int32)
        class_indices = gather_nd_simple(class_indices, indices)
        class_indices = torch.round(class_indices).type(torch.int32)
        
        axis_pred, angle_pred = separate_axis_from_angle(regression_rotation)
        axis_target, angle_target = separate_axis_from_angle(regression_target_rotation)
        
        #rotate the 3d model points with target and predicted rotations        
        #select model points according to the class indices
        selected_model_points = gather_torch(self.model_3d_points, class_indices, axis = 0)
        #expand dims of the rotation tensors to rotate all points along the dimension via broadcasting
        axis_pred = torch.unsqueeze(axis_pred, dim = 1)
        angle_pred = torch.unsqueeze(angle_pred, dim = 1)
        axis_target = torch.unsqueeze(axis_target, dim = 1)
        angle_target = torch.unsqueeze(angle_target, dim = 1)
        
        #also expand dims of the translation tensors to translate all points along the dimension via broadcasting
        regression_translation = torch.unsqueeze(regression_translation, dim = 1)
        regression_target_translation = torch.unsqueeze(regression_target_translation, dim = 1)
        
        transformed_points_pred = rotate(selected_model_points, axis_pred, angle_pred) + regression_translation
        transformed_points_target = rotate(selected_model_points, axis_target, angle_target) + regression_target_translation
        
        #distinct between symmetric and asymmetric objects
        sym_indices = torch.nonzero(torch.eq(is_symmetric, 1))
        asym_indices = torch.nonzero(torch.ne(is_symmetric, 1))
        
        sym_points_pred = gather_nd_simple(transformed_points_pred, sym_indices).reshape(-1, self.num_points, 3)
        asym_points_pred = gather_nd_simple(transformed_points_pred, asym_indices).reshape(-1, self.num_points, 3)
        
        sym_points_target = gather_nd_simple(transformed_points_target, sym_indices).reshape(-1, self.num_points, 3)
        asym_points_target = gather_nd_simple(transformed_points_target, asym_indices).reshape(-1, self.num_points, 3)
        
        # # compute transformed point distances
        sym_distances = calc_sym_distances(sym_points_pred, sym_points_target)
        asym_distances = calc_asym_distances(asym_points_pred, asym_points_target)

        distances = torch.cat([sym_distances, asym_distances], dim = 0)
        
        loss = torch.mean(distances)        
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)#in case of no annotations the loss is nan => replace with zero
        
        return loss


def separate_axis_from_angle(axis_angle_tensor):
    """ Separates the compact 3-dimensional axis_angle representation in the rotation axis and a rotation angle
        Args:
            axis_angle_tensor: tensor with a shape of 3 in the last dimension.
        Returns:
            axis: Tensor of the same shape as the input axis_angle_tensor but containing only the rotation axis and not the angle anymore
            angle: Tensor of the same shape as the input axis_angle_tensor except the last dimension is 1 and contains the rotation angle
        """
    squared = torch.square(axis_angle_tensor)
    summed = torch.sum(squared, dim = -1)
    angle = torch.unsqueeze(torch.sqrt(summed), dim = -1)
    
    axis = torch.div(axis_angle_tensor, angle)
    
    return axis, angle

def calc_sym_distances(sym_points_pred, sym_points_target):
    """ Calculates the average minimum point distance for symmetric objects
        Args:
            sym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            sym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average minimum point distance between both transformed 3D models
        """
    if sym_points_pred.shape[0] == 0:# 全是asym的情况
        return torch.Tensor([]).to(sym_points_pred.device)
    sym_points_pred = torch.unsqueeze(sym_points_pred, dim = 2)
    sym_points_target = torch.unsqueeze(sym_points_target, dim = 1)
    distances, _  = torch.min(torch.norm(sym_points_pred - sym_points_target, dim = -1), dim = -1)
    
    return torch.mean(distances, dim = -1)
    
def calc_asym_distances(asym_points_pred, asym_points_target):
    """ Calculates the average pairwise point distance for asymmetric objects
        Args:
            asym_points_pred: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the model's prediction
            asym_points_target: Tensor of shape (num_objects, num_3D_points, 3) containing all 3D model points transformed with the ground truth 6D pose
        Returns:
            Tensor of shape (num_objects) containing the average point distance between both transformed 3D models
        """
    distances = torch.norm(asym_points_pred - asym_points_target, dim = -1)
    
    return torch.mean(distances, dim = -1)


#copied and adapted the following functions from tensorflow graphics source because they did not work with unknown shape
#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def cross(vector1, vector2, name=None):
  """Computes the cross product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    vector2: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    axis: The dimension along which to compute the cross product.
    name: A name for this op which defaults to "vector_cross".
  Returns:
    A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
    represents the result of the cross product.
  """
  #with tf.compat.v1.name_scope(name, "vector_cross", [vector1, vector2]):
  vector1_x = vector1[:, :, 0]
  vector1_y = vector1[:, :, 1]
  vector1_z = vector1[:, :, 2]
  vector2_x = vector2[:, :, 0]
  vector2_y = vector2[:, :, 1]
  vector2_z = vector2[:, :, 2]
  n_x = vector1_y * vector2_z - vector1_z * vector2_y
  n_y = vector1_z * vector2_x - vector1_x * vector2_z
  n_z = vector1_x * vector2_y - vector1_y * vector2_x
  return torch.stack((n_x, n_y, n_z), dim = -1) # 在最后一维上stack

#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py
def dot(vector1, vector2, axis=-1, keepdims=True, name=None):
  """Computes the dot product between two tensors along an axis.
  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.
  Args:
    vector1: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    vector2: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    axis: The dimension along which to compute the dot product.
    keepdims: If True, retains reduced dimensions with length 1.
    name: A name for this op which defaults to "vector_dot".
  Returns:
    A tensor of shape `[A1, ..., Ai = 1, ..., An]`, where the dimension i = axis
    represents the result of the dot product.
  """
  #with tf.compat.v1.name_scope(name, "vector_dot", [vector1, vector2]):
  return torch.sum(input=vector1 * vector2, dim=axis, keepdim=keepdims)

#copied from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py
def rotate(point, axis, angle, name=None):
  r"""Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.
  Rotates a vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into a vector
  $$\mathbf{v}' \in {\mathbb{R}^3}$$ using the Rodrigues' rotation formula:
  $$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
  +\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    point: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point to rotate.
    axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a normalized axis.
    angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      represents an angle.
    name: A name for this op that defaults to "axis_angle_rotate".
  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
    a 3d point.
  Raises:
    ValueError: If `point`, `axis`, or `angle` are of different shape or if
    their respective shape is not supported.
  """
  #with tf.compat.v1.name_scope(name, "axis_angle_rotate", [point, axis, angle]):
  cos_angle = torch.cos(angle)
  axis_dot_point = dot(axis, point)
  return point * cos_angle + cross(axis, point) * torch.sin(angle) + axis * axis_dot_point * (1.0 - cos_angle)   
  
