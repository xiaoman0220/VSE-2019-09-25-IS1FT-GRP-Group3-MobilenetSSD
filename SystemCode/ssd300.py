
from __future__ import division
import numpy as np
import tensorflow as tf
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from collections import defaultdict
import warnings
import cv2
import random
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
import pickle
import math
from tqdm import tqdm
from tensorflow.python.ops import array_ops
import keras.layers as KL
from keras.models import Model
from keras.layers import Input, Lambda, Activation,Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate,BatchNormalization, Add, Conv2DTranspose
from keras.regularizers import l2
from keras import backend as K, initializers, regularizers, constraints
from keras.backend import image_data_format
from keras.backend.tensorflow_backend import _preprocess_conv2d_input, _preprocess_padding
from keras.engine.topology import InputSpec
from keras.layers import Conv2D
from keras.legacy.interfaces import conv2d_args_preprocessor, generate_legacy_interface
from keras.utils import conv_utils

try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")



def depthwise_conv2d_args_preprocessor(args, kwargs):
    converted = []
    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer'))
    args, kwargs, _converted = conv2d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted


legacy_depthwise_conv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=depthwise_conv2d_args_preprocessor)


class DepthwiseConv2D(Conv2D):

    # legacy_depthwise_conv2d_support = Solution.legacy_depthwise_conv2d_support
    @legacy_depthwise_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)

        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`SeparableConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.data_format is None:
            data_format = image_data_format()
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))

        x = _preprocess_conv2d_input(inputs, self.data_format)
        padding = _preprocess_padding(self.padding)
        strides = (1,) + self.strides + (1,)

        outputs = tf.nn.depthwise_conv2d(inputs, self.depthwise_kernel,
                                         strides=strides,
                                         padding=padding,
                                         rate=self.dilation_rate)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], self.filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, self.filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


DepthwiseConvolution2D = DepthwiseConv2D


class ssd_box_encode_decode_utils:
#     def __init__(self):
#         pass
    
    def convert_coordinates(self,tensor, start_index, conversion):
        ind = start_index
        tensor1 = np.copy(tensor).astype(np.float)
        if conversion == 'minmax2centroids':
            tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
            tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
            tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] # Set w
            tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] # Set h
        elif conversion == 'centroids2minmax':
            tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
            tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
            tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
            tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
        elif conversion == 'corners2centroids':
            tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
            tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
            tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] # Set w
            tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] # Set h
        elif conversion == 'centroids2corners':
            tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
            tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
            tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
            tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
        elif (conversion == 'minmax2corners') or (conversion == 'corners2minmax'):
            tensor1[..., ind+1] = tensor[..., ind+2]
            tensor1[..., ind+2] = tensor[..., ind+1]
        else:
            raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

        return tensor1
    def iou(self,boxes1, boxes2, coords='centroids'):
        if len(boxes1.shape) > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
        if len(boxes2.shape) > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

        if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
        if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

        if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

        if coords == 'centroids':
            # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
            boxes1 = ssd_box_encode_decode_utils.convert_coordinates(self,boxes1, start_index=0, conversion='centroids2minmax')
            boxes2 = ssd_box_encode_decode_utils.convert_coordinates(self,boxes2, start_index=0, conversion='centroids2minmax')
        elif not (coords in {'minmax', 'corners'}):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

        if coords in {'minmax', 'centroids'}:
            intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:,2]))
            union = (boxes1[:,1] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,2]) + (boxes2[:,1] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,2]) - intersection
        elif coords == 'corners':
            intersection = np.maximum(0, np.minimum(boxes1[:,2], boxes2[:,2]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,1], boxes2[:,1]))
            union = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1]) + (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1]) - intersection

        return intersection / union

    def greedy_nms(self,y_pred_decoded, iou_threshold=0.45, coords='corners'):
        y_pred_decoded_nms = []
        for batch_item in y_pred_decoded: # For the labels of each batch item...
            boxes_left = np.copy(batch_item)
            maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
            while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
                maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
                maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
                maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
                boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
                if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
                similarities = self.iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
                boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
            y_pred_decoded_nms.append(np.array(maxima))

        return y_pred_decoded_nms

    def _greedy_nms(self,predictions, iou_threshold=0.45, coords='corners'):
        
        boxes_left = np.copy(predictions)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = self.iou(boxes_left[:,1:], maximum_box[1:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        return np.array(maxima)

#     def _greedy_nms2(self, predictions, iou_threshold=0.45, coords='corners'):
#         boxes_left = np.copy(predictions)
#         maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
#         while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
#             maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
#             maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
#             maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
#             boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
#             if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
#             similarities = self.iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
#             boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
#         return np.array(maxima)

    def decode_y(y_pred,confidence_thresh=0.01,iou_threshold=0.45,top_k=200,input_coords='centroids',normalize_coords=True,img_height=None,img_width=None):
        ssd_box_encode_decode_utils_local = ssd_box_encode_decode_utils()
        if normalize_coords and ((img_height is None) or (img_width is None)):
            
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

        y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

        if input_coords == 'centroids':
            y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
            y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
            y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
            y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
            y_pred_decoded_raw = ssd_box_encode_decode_utils_local.convert_coordinates(tensor=y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
        elif input_coords == 'minmax':
            y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
            y_pred_decoded_raw = ssd_box_encode_decode_utils_local.convert_coordinates(tensor=y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
        elif input_coords == 'corners':
            y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        else:
            raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

        # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

        if normalize_coords:
            y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
            y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

        # 3: Apply confidence thresholding and non-maximum suppression per class

        n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

        y_pred_decoded = [] # Store the final predictions in this list
        


        for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
            pred = [] # Store the final predictions for this batch item here
            for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
                single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
                threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
                if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                    maxima = ssd_box_encode_decode_utils_local._greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners') # ...perform NMS on them.
                    maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                    maxima_output[:,0] = class_id # Write the class ID to the first column...
                    maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                    pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
            # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
            if pred: # If there are any predictions left after confidence-thresholding...
                pred = np.concatenate(pred, axis=0)
                if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                    top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                    pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
            y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

        return y_pred_decoded

    class SSDBoxEncoder:
        def __init__(self,
                     img_height,
                     img_width,
                     n_classes,
                     predictor_sizes,
                     min_scale=0.1,
                     max_scale=0.9,
                     scales=None,
                     aspect_ratios_global=[0.5, 1.0, 2.0],
                     aspect_ratios_per_layer=None,
                     two_boxes_for_ar1=True,
                     steps=None,
                     offsets=None,
                     limit_boxes=False,
                     variances=[1.0, 1.0, 1.0, 1.0],
                     pos_iou_threshold=0.5,
                     neg_iou_threshold=0.3,
                     coords='centroids',
                     normalize_coords=True):
            
            predictor_sizes = np.array(predictor_sizes)
            if len(predictor_sizes.shape) == 1:
                predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

            if (min_scale is None or max_scale is None) and scales is None:
                raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

            if scales:
                if (len(scales) != len(predictor_sizes)+1): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                    raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))
                scales = np.array(scales)
                if np.any(scales <= 0):
                    raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
            else: # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
                if not 0 < min_scale <= max_scale:
                    raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

            if not (aspect_ratios_per_layer is None):
                if (len(aspect_ratios_per_layer) != len(predictor_sizes)): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                    raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
                for aspect_ratios in aspect_ratios_per_layer:
                    if np.any(np.array(aspect_ratios) <= 0):
                        raise ValueError("All aspect ratios must be greater than zero.")
            else:
                if (aspect_ratios_global is None):
                    raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
                if np.any(np.array(aspect_ratios_global) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")

            if len(variances) != 4:
                raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
            variances = np.array(variances)
            if np.any(variances <= 0):
                raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

            if neg_iou_threshold > pos_iou_threshold:
                raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

            if not (coords == 'minmax' or coords == 'centroids' or coords == 'corners'):
                raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

            if (not (steps is None)) and (len(steps) != len(predictor_sizes)):
                raise ValueError("You must provide at least one step value per predictor layer.")

            if (not (offsets is None)) and (len(offsets) != len(predictor_sizes)):
                raise ValueError("You must provide at least one offset value per predictor layer.")

            self.img_height = img_height
            self.img_width = img_width
            self.n_classes = n_classes + 1
            self.predictor_sizes = predictor_sizes
            self.min_scale = min_scale
            self.max_scale = max_scale
            if (scales is None):
                self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
            else:
                # If a list of scales is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`.
                self.scales = scales
            if (aspect_ratios_per_layer is None):
                self.aspect_ratios = [aspect_ratios_global] * len(predictor_sizes)
            else:
                # If aspect ratios are given per layer, we'll use those.
                self.aspect_ratios = aspect_ratios_per_layer
            self.two_boxes_for_ar1 = two_boxes_for_ar1
            if (not steps is None):
                self.steps = steps
            else:
                self.steps = [None] * len(predictor_sizes)
            if (not offsets is None):
                self.offsets = offsets
            else:
                self.offsets = [None] * len(predictor_sizes)
            self.limit_boxes = limit_boxes
            self.variances = variances
            self.pos_iou_threshold = pos_iou_threshold
            self.neg_iou_threshold = neg_iou_threshold
            self.coords = coords
            self.normalize_coords = normalize_coords

            # Compute the number of boxes per cell.
            if aspect_ratios_per_layer:
                self.n_boxes = []
                for aspect_ratios in aspect_ratios_per_layer:
                    if (1 in aspect_ratios) & two_boxes_for_ar1:
                        self.n_boxes.append(len(aspect_ratios) + 1)
                    else:
                        self.n_boxes.append(len(aspect_ratios))
            else:
                if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                    self.n_boxes = len(aspect_ratios_global) + 1
                else:
                    self.n_boxes = len(aspect_ratios_global)

            self.boxes_list = [] # This will contain the anchor boxes for each predicotr layer.

            self.wh_list_diag = [] # Box widths and heights for each predictor layer
            self.steps_diag = [] # Horizontal and vertical distances between any two boxes for each predictor layer
            self.offsets_diag = [] # Offsets for each predictor layer
            self.centers_diag = [] # Anchor box center points as `(cy, cx)` for each predictor layer

            for i in range(len(self.predictor_sizes)):
                boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                       aspect_ratios=self.aspect_ratios[i],
                                                                                       this_scale=self.scales[i],
                                                                                       next_scale=self.scales[i+1],
                                                                                       this_steps=self.steps[i],
                                                                                       this_offsets=self.offsets[i],
                                                                                       diagnostics=True)

                print ('anchor boxes ', boxes.shape)
                self.boxes_list.append(boxes)
                self.wh_list_diag.append(wh)
                self.steps_diag.append(step)
                self.offsets_diag.append(offset)
                self.centers_diag.append(center)

        def generate_anchor_boxes_for_layer(self,
                                            feature_map_size,
                                            aspect_ratios,
                                            this_scale,
                                            next_scale,
                                            this_steps=None,
                                            this_offsets=None,
                                            diagnostics=False):
            size = min(self.img_height, self.img_width)
            # Compute the box widths and and heights for all aspect ratios
            wh_list = []
            for ar in aspect_ratios:
                if (ar == 1):
                    # Compute the regular anchor box for aspect ratio 1.
                    box_height = box_width = this_scale * size
                    wh_list.append((box_width, box_height))
                    if self.two_boxes_for_ar1:
                        # Compute one slightly larger version using the geometric mean of this scale value and the next.
                        box_height = box_width = np.sqrt(this_scale * next_scale) * size
                        wh_list.append((box_width, box_height))
                else:
                    box_width = this_scale * size * np.sqrt(ar)
                    box_height = this_scale * size / np.sqrt(ar)
                    wh_list.append((box_width, box_height))
            wh_list = np.array(wh_list)
            n_boxes = len(wh_list)

            # Compute the grid of box center points. They are identical for all aspect ratios.

            # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
            if (this_steps is None):
                step_height = self.img_height / feature_map_size[0]
                step_width = self.img_width / feature_map_size[1]
            else:
                if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                    step_height = this_steps[0]
                    step_width = this_steps[1]
                elif isinstance(this_steps, (int, float)):
                    step_height = this_steps
                    step_width = this_steps
            # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
            if (this_offsets is None):
                offset_height = 0.5
                offset_width = 0.5
            else:
                if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                    offset_height = this_offsets[0]
                    offset_width = this_offsets[1]
                elif isinstance(this_offsets, (int, float)):
                    offset_height = this_offsets
                    offset_width = this_offsets
            # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
            cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
            cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
            cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

            # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
            # where the last dimension will contain `(cx, cy, w, h)`
            boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

            boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
            boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
            boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
            boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

            # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
            boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='centroids2corners')

            # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
            if self.limit_boxes:
                x_coords = boxes_tensor[:,:,:,[0, 2]]
                x_coords[x_coords >= self.img_width] = self.img_width - 1
                x_coords[x_coords < 0] = 0
                boxes_tensor[:,:,:,[0, 2]] = x_coords
                y_coords = boxes_tensor[:,:,:,[1, 3]]
                y_coords[y_coords >= self.img_height] = self.img_height - 1
                y_coords[y_coords < 0] = 0
                boxes_tensor[:,:,:,[1, 3]] = y_coords

            # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
            if self.normalize_coords:
                boxes_tensor[:, :, :, [0, 2]] /= self.img_width
                boxes_tensor[:, :, :, [1, 3]] /= self.img_height

            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
            if self.coords == 'centroids':
                # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
                boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='corners2centroids')
            elif self.coords == 'minmax':
                # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
                boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='corners2minmax')

            if diagnostics:
                return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
            else:
                return boxes_tensor

        def generate_encode_template(self, batch_size, diagnostics=False):
            boxes_batch = []
            for boxes in self.boxes_list:
                # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
                # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
                boxes = np.expand_dims(boxes, axis=0)

                # print 'earlier box shape',boxes.shape


                # boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1),)
                boxes = np.repeat(boxes, batch_size,axis=0)

                # print 'after ', boxes.shape

                # Now reshape the 5D tensor above into a 3D tensor of shape
                # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
                # order of the tensor content will be identical to the order obtained from the reshaping operation
                # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
                # use the same default index order, which is C-like index ordering)

                # print 'box shape',boxes.shape
                boxes = np.reshape(boxes, (batch_size, -1, 4))
                boxes_batch.append(boxes)

            # Concatenate the anchor tensors from the individual layers to one.
            boxes_tensor = np.concatenate(boxes_batch, axis=1)

            # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
            #    It will contain all zeros for now, the classes will be set in the matching process that follows
            classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

            # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
            #    contains the same 4 variance values for every position in the last axis.
            variances_tensor = np.zeros_like(boxes_tensor)
            variances_tensor += self.variances # Long live broadcasting

            # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
            #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
            #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
            #    `boxes_tensor` a second time.
            y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

            if diagnostics:
                return y_encode_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
            else:
                return y_encode_template

        def encode_y(self, ground_truth_labels, diagnostics=False):

            # 1: Generate the template for y_encoded
            y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
            y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

            # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
            #    and for each matched box record the ground truth coordinates in `y_encoded`.
            #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

            class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

            for i in range(y_encode_template.shape[0]): # For each batch item...
                available_boxes = np.ones((y_encode_template.shape[1])) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
                negative_boxes = np.ones((y_encode_template.shape[1])) # 1 for all negative boxes, 0 otherwise
                for true_box in ground_truth_labels[i]: # For each ground truth box belonging to the current batch item...
                    true_box = true_box.astype(np.float)
                    if abs(true_box[3] - true_box[1] < 0.001) or abs(true_box[4] - true_box[2] < 0.001): continue # Protect ourselves against bad ground truth data: boxes with width or height equal to zero
                    if self.normalize_coords:
                        true_box[[1,3]] /= self.img_width # Normalize xmin and xmax to be within [0,1]
                        true_box[[2,4]] /= self.img_height # Normalize ymin and ymax to be within [0,1]
                    if self.coords == 'centroids':
                        true_box = ssd_box_encode_decode_utils.convert_coordinates(self,true_box, start_index=1, conversion='corners2centroids')
                    elif self.coords == 'minmax':
                        true_box = ssd_box_encode_decode_utils.convert_coordinates(self,true_box, start_index=1, conversion='corners2minmax')
                    similarities = ssd_box_encode_decode_utils.iou(self,y_encode_template[i,:,-12:-8], true_box[1:], coords=self.coords) # The iou similarities for all anchor boxes
                    negative_boxes[similarities >= self.neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                    similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                    available_and_thresh_met = np.copy(similarities)
                    available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                    assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                    if len(assign_indices) > 0: # If we have any matches
                        y_encoded[i,assign_indices,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                        available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                    else: # If we don't have any matches
                        best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                        y_encoded[i,best_match_index,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                        available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                        negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
                # Set the classes of all remaining available anchor boxes to class zero
                background_class_indices = np.nonzero(negative_boxes)[0]
                y_encoded[i,background_class_indices,0] = 1

            # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
            if self.coords == 'centroids':
                y_encoded[:,:,[-12,-11]] -= y_encode_template[:,:,[-12,-11]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
                y_encoded[:,:,[-12,-11]] /= y_encode_template[:,:,[-10,-9]] * y_encode_template[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
                y_encoded[:,:,[-10,-9]] /= y_encode_template[:,:,[-10,-9]] # w(gt) / w(anchor), h(gt) / h(anchor)
                y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encode_template[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
            elif self.coords == 'corners':
                y_encoded[:,:,-12:-8] -= y_encode_template[:,:,-12:-8] # (gt - anchor) for all four coordinates
                y_encoded[:,:,[-12,-10]] /= np.expand_dims(y_encode_template[:,:,-10] - y_encode_template[:,:,-12], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
                y_encoded[:,:,[-11,-9]] /= np.expand_dims(y_encode_template[:,:,-9] - y_encode_template[:,:,-11], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
                y_encoded[:,:,-12:-8] /= y_encode_template[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
            else:
                y_encoded[:,:,-12:-8] -= y_encode_template[:,:,-12:-8] # (gt - anchor) for all four coordinates
                y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encode_template[:,:,-11] - y_encode_template[:,:,-12], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
                y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encode_template[:,:,-9] - y_encode_template[:,:,-10], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
                y_encoded[:,:,-12:-8] /= y_encode_template[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

            if diagnostics:
                # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
                y_matched_anchors = np.copy(y_encoded)
                y_matched_anchors[:,:,-12:-8] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
                return y_encoded, y_matched_anchors
            else:
                return y_encoded

    def decode_y_debug(self,y_pred,confidence_thresh=0.01,iou_threshold=0.45,top_k=200,input_coords='centroids',normalize_coords=True,img_height=None,img_width=None,variance_encoded_in_target=False):
        if normalize_coords and ((img_height is None) or (img_width is None)):
            raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

        # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

        y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

        if input_coords == 'centroids':
            if variance_encoded_in_target:
                # Decode the predicted box center x and y coordinates.
                y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] + y_pred[:,:,[-8,-7]]
                # Decode the predicted box width and heigt.
                y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
            else:
                # Decode the predicted box center x and y coordinates.
                y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] * y_pred[:,:,[-4,-3]] + y_pred[:,:,[-8,-7]]
                # Decode the predicted box width and heigt.
                y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
            y_pred_decoded_raw = ssd_box_encode_decode_utils.convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
        elif input_coords == 'minmax':
            y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
            y_pred_decoded_raw = ssd_box_encode_decode_utils.convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
        elif input_coords == 'corners':
            y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
            y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
            y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
            y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        else:
            raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

        # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

        if normalize_coords:
            y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
            y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

        # 3: For each batch item, prepend each box's internal index to its coordinates.

        y_pred_decoded_raw2 = np.zeros((y_pred_decoded_raw.shape[0], y_pred_decoded_raw.shape[1], y_pred_decoded_raw.shape[2] + 1)) # Expand the last axis by one.
        y_pred_decoded_raw2[:,:,1:] = y_pred_decoded_raw
        y_pred_decoded_raw2[:,:,0] = np.arange(y_pred_decoded_raw.shape[1]) # Put the box indices as the first element for each box via broadcasting.
        y_pred_decoded_raw = y_pred_decoded_raw2

        # 4: Apply confidence thresholding and non-maximum suppression per class

        n_classes = y_pred_decoded_raw.shape[-1] - 5 # The number of classes is the length of the last axis minus the four box coordinates and minus the index

        y_pred_decoded = [] # Store the final predictions in this list
        for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
            pred = [] # Store the final predictions for this batch item here
            for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
                single_class = batch_item[:,[0, class_id + 1, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 6]` and...
                threshold_met = single_class[single_class[:,1] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
                if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                    maxima = self._greedy_nms_debug(threshold_met, iou_threshold=iou_threshold, coords='corners') # ...perform NMS on them.
                    maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                    maxima_output[:,0] = maxima[:,0] # Write the box index to the first column...
                    maxima_output[:,1] = class_id # ...and write the class ID to the second column...
                    maxima_output[:,2:] = maxima[:,1:] # ...and write the rest of the maxima data to the other columns...
                    pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
            # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
            pred = np.concatenate(pred, axis=0)
            if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,2], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
            y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

        return y_pred_decoded

    def _greedy_nms_debug(self,predictions, iou_threshold=0.45, coords='corners'):
        boxes_left = np.copy(predictions)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = self.iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        return np.array(maxima)

    def get_num_boxes_per_pred_layer(predictor_sizes, aspect_ratios, two_boxes_for_ar1):

        num_boxes_per_pred_layer = []
        for i in range(len(predictor_sizes)):
            if two_boxes_for_ar1:
                num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * (len(aspect_ratios[i]) + 1))
            else:
                num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * len(aspect_ratios[i]))
        return num_boxes_per_pred_layer

    def get_pred_layers(y_pred_decoded, num_boxes_per_pred_layer):
        pred_layers_all = []
        cum_boxes_per_pred_layer = np.cumsum(num_boxes_per_pred_layer)
        for batch_item in y_pred_decoded:
            pred_layers = []
            for prediction in batch_item:
                if (prediction[0] < 0) or (prediction[0] >= cum_boxes_per_pred_layer[-1]):
                    raise ValueError("Box index is out of bounds of the possible indices as given by the values in `num_boxes_per_pred_layer`.")
                for i in range(len(cum_boxes_per_pred_layer)):
                    if prediction[0] < cum_boxes_per_pred_layer[i]:
                        pred_layers.append(i)
                        break
            pred_layers_all.append(pred_layers)
        return pred_layers_all


class keras_layer_L2Normalization:
    def __init__(self):
        pass

    class L2Normalization(Layer):
        def __init__(self, gamma_init=20, **kwargs):
            if K.image_dim_ordering() == 'tf':
                self.axis = 3
            else:
                self.axis = 1
            self.gamma_init = gamma_init
            super(keras_layer_L2Normalization.L2Normalization, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            gamma = self.gamma_init * np.ones((input_shape[self.axis],))
            self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
            self.trainable_weights = [self.gamma]
            super(keras_layer_L2Normalization.L2Normalization, self).build(input_shape)

        def call(self, x, mask=None):
            output = K.l2_normalize(x, self.axis)
            return output * self.gamma

        def get_config(self):
            config = {
                'gamma_init': self.gamma_init
            }
            base_config = super(keras_layer_L2Normalization.L2Normalization, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))



class ssd_batch_generator:


    def _translate(self,image, horizontal=(0,40), vertical=(0,10)):
        rows,cols,ch = image.shape

        x = np.random.randint(horizontal[0], horizontal[1]+1)
        y = np.random.randint(vertical[0], vertical[1]+1)
        x_shift = random.choice([-x, x])
        y_shift = random.choice([-y, y])

        M = np.float32([[1,0,x_shift],[0,1,y_shift]])
        return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift

    def _flip(self,image, orientation='horizontal'):
        '''
        Flip the input image horizontally or vertically.
        '''
        if orientation == 'horizontal':
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)

    def _scale(self,image, min=0.9, max=1.1):

        rows,cols,ch = image.shape

        #Randomly select a scaling factor from the range passed.
        scale = np.random.uniform(min, max)

        M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
        return cv2.warpAffine(image, M, (cols, rows)), M, scale

    def _brightness(self,image, min=0.5, max=2.0):

        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

        random_br = np.random.uniform(min,max)

        #To protect against overflow: Calculate a mask for all pixels
        #where adjustment of the brightness would exceed the maximum
        #brightness value and set the value to the maximum at those pixels.
        mask = hsv[:,:,2] * random_br > 255
        v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
        hsv[:,:,2] = v_channel

        return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    def histogram_eq(self,image):
        '''
        Perform histogram equalization on the input image.

        See https://en.wikipedia.org/wiki/Histogram_equalization.
        '''

        image1 = np.copy(image)

        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)

        image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])

        image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

        return image1

    class BatchGenerator:

        def __init__(self,
                     box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'],
                     filenames=None,
                     filenames_type='text',
                     images_dir=None,
                     labels=None,
                     image_ids=None):
            
            self.box_output_format = box_output_format


            if not filenames is None:
                if isinstance(filenames, (list, tuple)):
                    self.filenames = filenames
                elif isinstance(filenames, str):
                    with open(filenames, 'rb') as f:
                        if filenames_type == 'pickle':
                            self.filenames = pickle.load(f)
                        elif filenames_type == 'text':
                            self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                        else:
                            raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
                else:
                    raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
            else:
                self.filenames = []

            if not labels is None:
                if isinstance(labels, str):
                    with open(labels, 'rb') as f:
                        self.labels = pickle.load(f)
                elif isinstance(labels, (list, tuple)):
                    self.labels = labels
                else:
                    raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
            else:
                self.labels = None

            if not image_ids is None:
                if isinstance(image_ids, str):
                    with open(image_ids, 'rb') as f:
                        self.image_ids = pickle.load(f)
                elif isinstance(image_ids, (list, tuple)):
                    self.image_ids = image_ids
                else:
                    raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
            else:
                self.image_ids = None

        def parse_csv(self,
                      images_dir,
                      labels_filename,
                      input_format,
                      include_classes='all',
                      random_sample=False,
                      ret=False):

            self.images_dir = images_dir
            self.labels_filename = labels_filename
            self.input_format = input_format
            self.include_classes = include_classes

            # Before we begin, make sure that we have a labels_filename and an input_format
            if self.labels_filename is None or self.input_format is None:
                raise ValueError("`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

            # Erase data that might have been parsed before
            self.filenames = []
            self.labels = []

            # First, just read in the CSV file lines and sort them.

            data = []

            with open(self.labels_filename, newline='') as csvfile:
                csvread = csv.reader(csvfile, delimiter=',')
                next(csvread) # Skip the header row.
                for row in csvread: # For every line (i.e for every bounding box) in the CSV file...
                    if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes: # If the class_id is among the classes that are to be included in the dataset...
                        box = [] # Store the box class and coordinates here
                        box.append(row[self.input_format.index('image_name')].strip()) # Select the image name column in the input format and append its content to `box`
                        for element in self.box_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                            box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                        data.append(box)

            data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

            # Now that we've made sure that the data is sorted by file names,
            # we can compile the actual samples and labels lists

            current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
            current_labels = [] # The list where we collect all ground truth boxes for a given image
            add_to_dataset = False
            for i, box in enumerate(data):

                if box[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                    current_labels.append(box[1:])
                    if i == len(data)-1: # If this is the last line of the CSV file
                        if random_sample: # In case we're not using the full dataset, but a random sample of it.
                            p = np.random.uniform(0,1)
                            if p >= (1-random_sample):
                                self.labels.append(np.stack(current_labels, axis=0))
                                self.filenames.append(os.path.join(self.images_dir, current_file))
                        else:
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                else: # If this box belongs to a new image file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                    current_labels = [] # Reset the labels list because this is a new file.
                    current_file = box[0]
                    current_labels.append(box[1:])
                    if i == len(data)-1: # If this is the last line of the CSV file
                        if random_sample: # In case we're not using the full dataset, but a random sample of it.
                            p = np.random.uniform(0,1)
                            if p >= (1-random_sample):
                                self.labels.append(np.stack(current_labels, axis=0))
                                self.filenames.append(os.path.join(self.images_dir, current_file))
                        else:
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))

            if ret: # In case we want to return these
                return self.filenames, self.labels

        def parse_xml(self,
                      images_dirs,
                      image_set_filenames,
                      annotations_dirs=[],
                      classes=['background','Input','IP','OP','Output'],
                      include_classes = 'all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False,
                      percent=1):

            self.images_dirs = images_dirs
            self.annotations_dirs = annotations_dirs
            self.image_set_filenames = image_set_filenames
            self.classes = classes
            # print (self.classes)
            self.include_classes = include_classes

            # Erase data that might have been parsed before.
            self.filenames = []
            self.image_ids = []
            self.labels = []
            if not annotations_dirs:
                self.labels = None
                annotations_dirs = [None] * len(images_dirs)

            for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
                # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
                with open(image_set_filename) as f:
                    image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                    self.image_ids += image_ids
                
                #control the percentage of dataset used
                length = int(math.ceil((len(image_ids))*percent))
                print(length)
    #             self.image_ids = image_ids[:length-1]
                
                # Loop over all images in this dataset.
                #for image_id in image_ids:
                for image_id in tqdm(image_ids, desc=os.path.basename(image_set_filename)):

                    filename = '{}'.format(image_id) + '.jpg'
                    self.filenames.append(os.path.join(images_dir, filename))

                    if not annotations_dir is None:
                        # Parse the XML file for this image.
                        with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                            soup = BeautifulSoup(f, 'xml')

                        folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                        #filename = soup.filename.text

                        boxes = [] # We'll store all boxes for this image here
                        objects = soup.find_all('object') # Get a list of all objects in this image



                        # Parse the data for each object
                        for obj in objects:
                            class_name = obj.find('name').text
                            # print class_name
                            class_id = self.classes.index(class_name)
                            # Check if this class is supposed to be included in the dataset
                            if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                            pose = obj.pose.text
                            truncated = int(obj.truncated.text)
                            if exclude_truncated and (truncated == 1): continue
                            difficult = int(obj.difficult.text)
                            if exclude_difficult and (difficult == 1): continue
                            xmin = int(obj.bndbox.xmin.text)
                            ymin = int(obj.bndbox.ymin.text)
                            xmax = int(obj.bndbox.xmax.text)
                            ymax = int(obj.bndbox.ymax.text)


                            item_dict = {'folder': folder,
                                         'image_name': filename,
                                         'image_id': image_id,
                                         'class_name': class_name,
                                         'class_id': class_id,
                                         'pose': pose,
                                         'truncated': truncated,
                                         'difficult': difficult,
                                         'xmin': xmin,
                                         'ymin': ymin,
                                         'xmax': xmax,
                                         'ymax': ymax}
                            box = []
                            for item in self.box_output_format:
                                box.append(item_dict[item])
                            boxes.append(box)

                        # print 'size of boxex',len(boxes)

                        self.labels.append(boxes)

            if ret:
                return self.filenames, self.labels, self.image_ids

        def parse_darknet(self,
                      images_dirs,
                      image_set_filenames,
                      annotations_dirs=[],
                      classes=['background','Input','IP','OP','Output'],
                      include_classes = 'all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False, 
                      map_dict={},
                      label_string="bike_person_merged_labels"):
            
            # Set class members.
            self.images_dirs = images_dirs
            self.annotations_dirs = annotations_dirs
            self.image_set_filenames = image_set_filenames
            self.classes = classes
            # print (self.classes)
            self.include_classes = include_classes

            # Erase data that might have been parsed before.
            self.filenames = []
            self.image_ids = []
            self.labels = []
            if not annotations_dirs:
                self.labels = None
                annotations_dirs = [None] * len(images_dirs)

            
                # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filenames[0]) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

                # Loop over all images in this dataset.
                #for image_id in image_ids:
            for image_id in image_ids:
                self.filenames.append(image_id.strip())

            #image_id = image_id.split(".")[0]
                #image_id = image_id.split(os.sep)[-1]

                # print image_id
                img = cv2.imread(image_id)
                if img is None:
                    print(image_id.strip())
                    continue
                W, H = img.shape[1], img.shape[0]

                
                # Parse the XML file for this image.
                label_path = image_id.replace("ssd_images", label_string)
                label_path = label_path.replace(".jpg", ".txt").replace(".png", ".txt")
                label_file = os.path.join(label_path)
                #print 'label file: ', label_file
                lines = open(label_file, 'r').readlines()
                boxes = [] # We'll store all boxes for this image here
                for l in lines:
                    idx, xc, yc, w, h = l.strip().split()
                    idx, xc, yc, w, h = int(idx), float(xc), float(yc), float(w), float(h)
                    xmin = int((xc - w/2.0)*W)
                    xmax = int((xc + w/2.0)*W)
                    ymin = int((yc - h/2.0)*H)
                    ymax = int((yc + h/2.0)*H)

                    idx = map_dict[idx]
                    # print idx
                    boxes.append([idx, xmin, ymin, xmax, ymax])
                # for b in boxes:
                #     cv2.rectangle(img, (b[1], b[2]), (b[3], b[4]), (0,255,0))
                # cv2.imshow("boxes", img)
                # cv2.waitKey(0)
                self.labels.append(boxes)
            if ret:
                return self.filenames, self.labels, self.image_ids

        def parse_json(self,
                       images_dirs,
                       annotations_filenames,
                       ground_truth_available=False,
                       include_classes = 'all',
                       ret=False):
            
            self.images_dirs = images_dirs
            self.annotations_filenames = annotations_filenames
            self.include_classes = include_classes
            # Erase data that might have been parsed before.
            self.filenames = []
            self.image_ids = []
            self.labels = []
            if not ground_truth_available:
                self.labels = None

            # Build the dictionaries that map between class names and class IDs.
            with open(annotations_filenames[0], 'r') as f:
                annotations = json.load(f)
            # Unfortunately the 80 MS COCO class IDs are not all consecutive. They go
            # from 1 to 90 and some numbers are skipped. Since the IDs that we feed
            # into a neural network must be consecutive, we'll save both the original
            # (non-consecutive) IDs as well as transformed maps.
            # We'll save both the map between the original
            self.cats_to_names = {} # The map between class names (values) and their original IDs (keys)
            self.classes_to_names = [] # A list of the class names with their indices representing the transformed IDs
            self.classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
            self.cats_to_classes = {} # A dictionary that maps between the original (keys) and the transformed IDs (values)
            self.classes_to_cats = {} # A dictionary that maps between the transformed (keys) and the original IDs (values)
            for i, cat in enumerate(annotations['categories']):
                self.cats_to_names[cat['id']] = cat['name']
                self.classes_to_names.append(cat['name'])
                self.cats_to_classes[cat['id']] = i + 1
                self.classes_to_cats[i + 1] = cat['id']

            # Iterate over all datasets.
            for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
                # Load the JSON file.
                with open(annotations_filename, 'r') as f:
                    annotations = json.load(f)

                if ground_truth_available:
                    # Create the annotations map, a dictionary whose keys are the image IDs
                    # and whose values are the annotations for the respective image ID.
                    image_ids_to_annotations = defaultdict(list)
                    for annotation in annotations['annotations']:
                        image_ids_to_annotations[annotation['image_id']].append(annotation)

                # Iterate over all images in the dataset.
                for img in annotations['images']:

                    self.filenames.append(os.path.join(images_dir, img['file_name']))
                    self.image_ids.append(img['id'])

                    if ground_truth_available:
                        # Get all annotations for this image.
                        annotations = image_ids_to_annotations[img['id']]
                        boxes = []
                        for annotation in annotations:
                            cat_id = annotation['category_id']
                            # Check if this class is supposed to be included in the dataset.
                            if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                            # Transform the original class ID to fit in the sequence of consecutive IDs.
                            class_id = self.cats_to_classes[cat_id]
                            xmin = annotation['bbox'][0]
                            ymin = annotation['bbox'][1]
                            width = annotation['bbox'][2]
                            height = annotation['bbox'][3]
                            # Compute `xmax` and `ymax`.
                            xmax = xmin + width
                            ymax = ymin + height
                            item_dict = {'image_name': img['file_name'],
                                         'image_id': img['id'],
                                         'class_id': class_id,
                                         'xmin': xmin,
                                         'ymin': ymin,
                                         'xmax': xmax,
                                         'ymax': ymax}
                            box = []
                            for item in self.box_output_format:
                                box.append(item_dict[item])
                            boxes.append(box)
                        self.labels.append(boxes)

            if ret:
                return self.filenames, self.labels, self.image_ids

        def save_filenames_and_labels(self, filenames_path='filenames.pkl', labels_path=None, image_ids_path=None):
        
            with open(filenames_path, 'wb') as f:
                pickle.dump(self.filenames, f)
            if not labels_path is None:
                with open(labels_path, 'wb') as f:
                    pickle.dump(self.labels, f)
            if not image_ids_path is None:
                with open(image_ids_path, 'wb') as f:
                    pickle.dump(self.image_ids, f)

        def generate(self,
                     batch_size=32,
                     shuffle=True,
                     train=True,
                     ssd_box_encoder=None,
                     returns={'processed_images', 'encoded_labels'},
                     convert_to_3_channels=True,
                     equalize=False,
                     brightness=False,
                     flip=False,
                     translate=False,
                     scale=False,
                     max_crop_and_resize=False,
                     random_pad_and_resize=False,
                     random_crop=False,
                     crop=False,
                     resize=False,
                     gray=False,
                     limit_boxes=True,
                     include_thresh=0.3,
                     subtract_mean=None,
                     divide_by_stddev=None,
                     swap_channels=False,
                     keep_images_without_gt=False):
            

            if shuffle: # Shuffle the data before we begin
                if (self.labels is None) and (self.image_ids is None):
                    self.filenames = sklearn.utils.shuffle(self.filenames)
                elif (self.labels is None):
                    self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
                elif (self.image_ids is None):
                    self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
                else:
                    self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)
            current = 0

            # Find out the indices of the box coordinates in the label data
            xmin = self.box_output_format.index('xmin')
            ymin = self.box_output_format.index('ymin')
            xmax = self.box_output_format.index('xmax')
            ymax = self.box_output_format.index('ymax')
            ios = np.amin([xmin, ymin, xmax, ymax]) # Index offset, we need this for the inverse coordinate transform indices.

            while True:

                batch_X, batch_y = [], []

                if current >= len(self.filenames):
                    current = 0
                    if shuffle:
                        # Shuffle the data after each complete pass
                        if (self.labels is None) and (self.image_ids is None):
                            self.filenames = sklearn.utils.shuffle(self.filenames)
                        elif (self.labels is None):
                            self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
                        elif (self.image_ids is None):
                            self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
                        else:
                            self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)

                # Get the image filepaths for this batch.
                batch_filenames = self.filenames[current:current+batch_size]

                # Load the images for this batch.
                for filename in batch_filenames:
                    with Image.open(filename) as img:
                        batch_X.append(np.array(img))

                # Get the labels for this batch (if there are any).
                if not (self.labels is None):
                    batch_y = deepcopy(self.labels[current:current+batch_size])
                else:
                    batch_y = None

                # Get the image IDs for this batch (if there are any).
                if not self.image_ids is None:
                    batch_image_ids = self.image_ids[current:current+batch_size]
                else:
                    batch_image_ids = None

                # Create the array that is to contain the inverse coordinate transformation values for this batch.
                batch_inverse_coord_transform = np.array([[[0, 1]] * 4] * batch_size, dtype=np.float) # Array of shape `(batch_size, 4, 2)`, where the last axis contains an additive and a multiplicative scalar transformation constant.

                if 'original_images' in returns:
                    batch_original_images = deepcopy(batch_X) # The original, unaltered images
                if 'original_labels' in returns and not batch_y is None:
                    batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

                current += batch_size

                batch_items_to_remove = [] # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list.

                for i in range(len(batch_X)):

                    img_height, img_width = batch_X[i].shape[0], batch_X[i].shape[1]

                    if not batch_y is None:
                        # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                        if (len(batch_y[i]) == 0) and not keep_images_without_gt:
                            batch_items_to_remove.append(i)
                        # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks.
                        batch_y[i] = np.array(batch_y[i])

                    # From here on, perform some optional image transformations.

                    if (batch_X[i].ndim == 2):
                        if convert_to_3_channels:
                            # Convert the 1-channel image into a 3-channel image.
                            batch_X[i] = np.stack([batch_X[i]] * 3, axis=-1)
                        else:
                            # batch_X[i].ndim must always be 3, even for single-channel images.
                            batch_X[i] = np.expand_dims(batch_X[i], axis=-1)

                    if equalize:
                        batch_X[i] = ssd_batch_generator.histogram_eq(self,batch_X[i])

                    if brightness:
                        p = np.random.uniform(0,1)
                        if p >= (1-brightness[2]):
                            batch_X[i] = ssd_batch_generator._brightness(self,batch_X[i], min=brightness[0], max=brightness[1])

                    if flip: # Performs flips along the vertical axis only (i.e. horizontal flips).
                        p = np.random.uniform(0,1)
                        if p >= (1-flip):
                            batch_X[i] = ssd_batch_generator._flip(self,image=batch_X[i])
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                batch_y[i][:,[xmin,xmax]] = img_width - batch_y[i][:,[xmax,xmin]] # xmin and xmax are swapped when mirrored

                    if translate:
                        p = np.random.uniform(0,1)
                        if p >= (1-translate[2]):
                            # Translate the image and return the shift values so that we can adjust the labels
                            batch_X[i], xshift, yshift = ssd_batch_generator._translate(batch_X[i], translate[0], translate[1])
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Adjust the box coordinates.
                                batch_y[i][:,[xmin,xmax]] += xshift
                                batch_y[i][:,[ymin,ymax]] += yshift
                                # Limit the box coordinates to lie within the image boundaries
                                if limit_boxes:
                                    before_limiting = deepcopy(batch_y[i])
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                    # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                    # process that they don't serve as useful training examples anymore, because too little of them is
                                    # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                    # `include_thresh` of the box area before limiting.
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                    if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                    if scale:
                        p = np.random.uniform(0,1)
                        if p >= (1-scale[2]):
                            # Rescale the image and return the transformation matrix M so we can use it to adjust the box coordinates
                            batch_X[i], M, scale_factor = ssd_batch_generator._scale(batch_X[i], scale[0], scale[1])
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Adjust the box coordinates.
                                # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                                toplefts = np.array([batch_y[i][:,xmin], batch_y[i][:,ymin], np.ones(batch_y[i].shape[0])])
                                bottomrights = np.array([batch_y[i][:,xmax], batch_y[i][:,ymax], np.ones(batch_y[i].shape[0])])
                                new_toplefts = (np.dot(M, toplefts)).T
                                new_bottomrights = (np.dot(M, bottomrights)).T
                                batch_y[i][:,[xmin,ymin]] = new_toplefts.astype(np.int)
                                batch_y[i][:,[xmax,ymax]] = new_bottomrights.astype(np.int)
                                # Limit the box coordinates to lie within the image boundaries
                                if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                                    before_limiting = deepcopy(batch_y[i])
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                    # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                    # process that they don't serve as useful training examples anymore, because too little of them is
                                    # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                    # `include_thresh` of the box area before limiting.
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                    if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                    if max_crop_and_resize:
                        # The ratio of the two aspect ratios (source image and target size) determines the maximal possible crop.
                        image_aspect_ratio = img_width / img_height
                        resize_aspect_ratio = max_crop_and_resize[1] / max_crop_and_resize[0]

                        if image_aspect_ratio < resize_aspect_ratio:
                            crop_width = img_width
                            crop_height = int(round(crop_width / resize_aspect_ratio))
                        else:
                            crop_height = img_height
                            crop_width = int(round(crop_height * resize_aspect_ratio))
                        # The actual cropping and resizing will be done by the random crop and resizing operations below.
                        # Here, we only set the parameters for them.
                        random_crop = (crop_height, crop_width, max_crop_and_resize[2], max_crop_and_resize[3])
                        resize = (max_crop_and_resize[0], max_crop_and_resize[1])

                    if random_pad_and_resize:

                        resize_aspect_ratio = random_pad_and_resize[1] / random_pad_and_resize[0]

                        if img_width < img_height:
                            crop_height = img_height
                            crop_width = int(round(crop_height * resize_aspect_ratio))
                        else:
                            crop_width = img_width
                            crop_height = int(round(crop_width / resize_aspect_ratio))
                        # The actual cropping and resizing will be done by the random crop and resizing operations below.
                        # Here, we only set the parameters for them.
                        if max_crop_and_resize:
                            p = np.random.uniform(0,1)
                            if p >= (1-random_pad_and_resize[4]):
                                random_crop = (crop_height, crop_width, random_pad_and_resize[2], random_pad_and_resize[3])
                                resize = (random_pad_and_resize[0], random_pad_and_resize[1])
                        else:
                            random_crop = (crop_height, crop_width, random_pad_and_resize[2], random_pad_and_resize[3])
                            resize = (random_pad_and_resize[0], random_pad_and_resize[1])

                    if random_crop:
                        # Compute how much room we have in both dimensions to make a random crop.
                        # A negative number here means that we want to crop out a patch that is larger than the original image in the respective dimension,
                        # in which case we will create a black background canvas onto which we will randomly place the image.
                        y_range = img_height - random_crop[0]
                        x_range = img_width - random_crop[1]
                        # Keep track of the number of trials and of whether or not the most recent crop contains at least one object
                        min_1_object_fulfilled = False
                        trial_counter = 0
                        while (not min_1_object_fulfilled) and (trial_counter < random_crop[3]):
                            # Select a random crop position from the possible crop positions
                            if y_range >= 0: crop_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension
                            else: crop_ymin = np.random.randint(0, -y_range + 1) # The possible positions for the image on the background canvas in the vertical dimension
                            if x_range >= 0: crop_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension
                            else: crop_xmin = np.random.randint(0, -x_range + 1) # The possible positions for the image on the background canvas in the horizontal dimension
                            # Perform the crop
                            if y_range >= 0 and x_range >= 0: # If the patch to be cropped out is smaller than the original image in both dimenstions, we just perform a regular crop
                                # Crop the image
                                patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0], crop_xmin:crop_xmin+random_crop[1]])
                                # Add the parameters to reverse this transformation.
                                patch_y_inverse_y = crop_ymin
                                patch_y_inverse_x = crop_xmin
                                if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                    # Translate the box coordinates into the new coordinate system: Cropping shifts the origin by `(crop_ymin, crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] -= crop_ymin
                                    patch_y[:,[xmin,xmax]] -= crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Both the x- and y-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        y_coords = patch_y[:,[ymin,ymax]]
                                        y_coords[y_coords < 0] = 0
                                        y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                        patch_y[:,[ymin,ymax]] = y_coords
                                        x_coords = patch_y[:,[xmin,xmax]]
                                        x_coords[x_coords < 0] = 0
                                        x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                        patch_y[:,[xmin,xmax]] = x_coords
                            elif y_range >= 0 and x_range < 0: # If the crop is larger than the original image in the horizontal dimension only,...
                                # Crop the image
                                patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0]]) # ...crop the vertical dimension just as before,...
                                canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                canvas[:, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                                patch_X = canvas
                                # Add the parameters to reverse this transformation.
                                patch_y_inverse_y = crop_ymin
                                patch_y_inverse_x = -crop_xmin
                                if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(crop_ymin, -crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] -= crop_ymin
                                    patch_y[:,[xmin,xmax]] += crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Only the y-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        y_coords = patch_y[:,[ymin,ymax]]
                                        y_coords[y_coords < 0] = 0
                                        y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                        patch_y[:,[ymin,ymax]] = y_coords
                            elif y_range < 0 and x_range >= 0: # If the crop is larger than the original image in the vertical dimension only,...
                                # Crop the image
                                patch_X = np.copy(batch_X[i][:,crop_xmin:crop_xmin+random_crop[1]]) # ...crop the horizontal dimension just as in the first case,...
                                canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                canvas[crop_ymin:crop_ymin+img_height, :] = patch_X # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                                patch_X = canvas
                                # Add the parameters to reverse this transformation.
                                patch_y_inverse_y = -crop_ymin
                                patch_y_inverse_x = crop_xmin
                                if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] += crop_ymin
                                    patch_y[:,[xmin,xmax]] -= crop_xmin
                                    # Limit the box coordinates to lie within the new image boundaries
                                    if limit_boxes:
                                        # Only the x-coordinates might need to be limited
                                        before_limiting = np.copy(patch_y)
                                        x_coords = patch_y[:,[xmin,xmax]]
                                        x_coords[x_coords < 0] = 0
                                        x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                        patch_y[:,[xmin,xmax]] = x_coords
                            else:  # If the crop is larger than the original image in both dimensions,...
                                patch_X = np.copy(batch_X[i])
                                canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                                canvas[crop_ymin:crop_ymin+img_height, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                                patch_X = canvas
                                # Add the parameters to reverse this transformation.
                                patch_y_inverse_y = -crop_ymin
                                patch_y_inverse_x = -crop_xmin
                                if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                    # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, -crop_xmin)`
                                    patch_y = np.copy(batch_y[i])
                                    patch_y[:,[ymin,ymax]] += crop_ymin
                                    patch_y[:,[xmin,xmax]] += crop_xmin
                                    # Note that no limiting is necessary in this case
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                # process that they don't serve as useful training examples anymore, because too little of them is
                                # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                # `include_thresh` of the box area before limiting.
                                if limit_boxes and (y_range >= 0 or x_range >= 0):
                                    before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                    after_area = (patch_y[:,xmax] - patch_y[:,xmin]) * (patch_y[:,ymax] - patch_y[:,ymin])
                                    if include_thresh == 0: patch_y = patch_y[after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                    else: patch_y = patch_y[after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all
                                trial_counter += 1 # We've just used one of our trials
                                # Check if we have found a valid crop
                                if random_crop[2] == 0: # If `min_1_object == 0`, break out of the while loop after the first loop because we are fine with whatever crop we got
                                    batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                    batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                    batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                                    batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                                    break
                                elif len(patch_y) > 0: # If we have at least one object left, this crop is valid and we can stop
                                    min_1_object_fulfilled = True
                                    batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                    batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                    batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                                    batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                                elif (trial_counter >= random_crop[3]) and (not i in batch_items_to_remove): # If we've reached the trial limit and still not found a valid crop, remove this image from the batch
                                    batch_items_to_remove.append(i)
                            else: # If `batch_y` is `None`, i.e. if we don't have ground truth data, any crop is a valid crop.
                                batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                                batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                                break
                        # Update the image size so that subsequent transformations can work correctly.
                        img_height = random_crop[0]
                        img_width = random_crop[1]

                    if crop:
                        # Crop the image
                        batch_X[i] = np.copy(batch_X[i][crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])
                        # Update the image size so that subsequent transformations can work correctly
                        img_height -= crop[0] + crop[1]
                        img_width -= crop[2] + crop[3]
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            # Translate the box coordinates into the new coordinate system if necessary: The origin is shifted by `(crop[0], crop[2])` (i.e. by the top and left crop values)
                            # If nothing was cropped off from the top or left of the image, the coordinate system stays the same as before
                            if crop[0] > 0:
                                batch_y[i][:,[ymin,ymax]] -= crop[0]
                            if crop[2] > 0:
                                batch_y[i][:,[xmin,xmax]] -= crop[2]
                            # Limit the box coordinates to lie within the new image boundaries
                            if limit_boxes:
                                before_limiting = np.copy(batch_y[i])
                                # We only need to check those box coordinates that could possibly have been affected by the cropping
                                # For example, if we only crop off the top and/or bottom of the image, there is no need to check the x-coordinates
                                if crop[0] > 0:
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords < 0] = 0
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                if crop[1] > 0:
                                    y_coords = batch_y[i][:,[ymin,ymax]]
                                    y_coords[y_coords >= img_height] = img_height - 1
                                    batch_y[i][:,[ymin,ymax]] = y_coords
                                if crop[2] > 0:
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords < 0] = 0
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                if crop[3] > 0:
                                    x_coords = batch_y[i][:,[xmin,xmax]]
                                    x_coords[x_coords >= img_width] = img_width - 1
                                    batch_y[i][:,[xmin,xmax]] = x_coords
                                # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                # process that they don't serve as useful training examples anymore, because too little of them is
                                # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                # `include_thresh` of the box area before limiting.
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                    if resize:
                        batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                        batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],1] *= (img_height / resize[0])
                        batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],1] *= (img_width / resize[1])
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            batch_y[i][:,[ymin,ymax]] = batch_y[i][:,[ymin,ymax]] * (resize[0] / img_height)
                            batch_y[i][:,[xmin,xmax]] = batch_y[i][:,[xmin,xmax]] * (resize[1] / img_width)
                        img_width, img_height = resize # Updating these at this point is unnecessary, but it's one fewer source of error if this method gets expanded in the future.

                    if gray:
                        batch_X[i] = cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY)
                        if convert_to_3_channels:
                            batch_X[i] = np.stack([batch_X[i]] * 3, axis=-1)
                        else:
                            batch_X[i] = np.expand_dims(batch_X[i], axis=-1)

                # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes.
                #          At this point, all images must have the same size, otherwise you will get an error during training.
                batch_X = np.array(batch_X)

                if not keep_images_without_gt:
                    # If any batch items need to be removed because of failed random cropping, remove them now.
                    batch_inverse_coord_transform = np.delete(batch_inverse_coord_transform, batch_items_to_remove, axis=0)
                    batch_X = np.delete(batch_X, batch_items_to_remove, axis=0)
                    for j in sorted(batch_items_to_remove, reverse=True):
                        # This isn't efficient, but it hopefully should not need to be done often anyway.
                        batch_filenames.pop(j)
                        if not batch_y is None: batch_y.pop(j)
                        if not batch_image_ids is None: batch_image_ids.pop(j)
                        if 'original_images' in returns: batch_original_images.pop(j)
                        if 'original_labels' in returns and not batch_y is None: batch_original_labels.pop(j)

                # Perform image transformations that can be bulk-applied to the whole batch.
                if not (subtract_mean is None):
                    batch_X = batch_X.astype(np.int16) - np.array(subtract_mean)
                if not (divide_by_stddev is None):
                    batch_X = batch_X.astype(np.int16) / np.array(divide_by_stddev)
                if swap_channels:
                    batch_X = batch_X[:,:,:,[2, 1, 0]]

                if train: # During training we need the encoded labels instead of the format that `batch_y` has
                    if ssd_box_encoder is None:
                        raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                    if 'matched_anchors' in returns:
                        batch_y_true, batch_matched_anchors = ssd_box_encoder.encode_y(batch_y, diagnostics=True) # Encode the labels into the `y_true` tensor that the SSD loss function needs.
                    else:
                        batch_y_true = ssd_box_encoder.encode_y(batch_y, diagnostics=False) # Encode the labels into the `y_true` tensor that the SSD loss function needs.

                # Compile the output.
                ret = []
                ret.append(batch_X)
                if train:
                    ret.append(batch_y_true)
                    if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
                if 'processed_labels' in returns and not batch_y is None: ret.append(batch_y)
                if 'filenames' in returns: ret.append(batch_filenames)
                if 'image_ids' in returns and not batch_image_ids is None: ret.append(batch_image_ids)
                if 'inverse_transform' in returns: ret.append(batch_inverse_coord_transform)
                if 'original_images' in returns: ret.append(batch_original_images)
                if 'original_labels' in returns and not batch_y is None: ret.append(batch_original_labels)

                yield ret

        def get_filenames_labels(self):
            '''
            Returns:
                The list of filenames, the list of labels, and the list of image IDs.
            '''
            return self.filenames, self.labels, self.image_ids

        def get_n_samples(self):
            '''
            Returns:
                The number of image files in the initialized dataset.
            '''
            return len(self.filenames)

        def process_offline(self,
                            dest_path='',
                            start=0,
                            stop='all',
                            crop=False,
                            equalize=False,
                            brightness=False,
                            flip=False,
                            translate=False,
                            scale=False,
                            resize=False,
                            gray=False,
                            limit_boxes=True,
                            include_thresh=0.3,
                            diagnostics=False):

            import gc

            targets_for_csv = []
            if stop == 'all':
                stop = len(self.filenames)

            if diagnostics:
                processed_images = []
                original_images = []
                processed_labels = []

            # Find out the indices of the box coordinates in the label data
            xmin = self.box_output_format.index('xmin')
            xmax = self.box_output_format.index('xmax')
            ymin = self.box_output_format.index('ymin')
            ymax = self.box_output_format.index('ymax')

            for k, filename in enumerate(self.filenames[start:stop]):
                i = k + start
                with Image.open('{}'.format(os.path.join(self.images_path, filename))) as img:
                    image = np.array(img)
                targets = np.copy(self.labels[i])

                if diagnostics:
                    original_images.append(image)

                img_height, img_width, ch = image.shape

                if equalize:
                    image = ssd_batch_generator.histogram_eq(self,image)

                if brightness:
                    p = np.random.uniform(0,1)
                    if p >= (1-brightness[2]):
                        image = ssd_batch_generator._brightness(self,image, min=brightness[0], max=brightness[1])

                # Could easily be extended to also allow vertical flipping, but I'm not convinced of the
                # usefulness of vertical flipping either empirically or theoretically, so I'm going for simplicity.
                # If you want to allow vertical flipping, just change this function to pass the respective argument
                # to `_flip()`.
                if flip:
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        image = ssd_batch_generator._flip(self,image)
                        targets[:,[0,1]] = img_width - targets[:,[1,0]] # xmin and xmax are swapped when mirrored

                if translate:
                    p = np.random.uniform(0,1)
                    if p >= (1-translate[2]):
                        image, xshift, yshift = ssd_batch_generator._translate(self,image, translate[0], translate[1])
                        targets[:,[0,1]] += xshift
                        targets[:,[2,3]] += yshift
                        if limit_boxes:
                            before_limiting = np.copy(targets)
                            x_coords = targets[:,[0,1]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            targets[:,[0,1]] = x_coords
                            y_coords = targets[:,[2,3]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            targets[:,[2,3]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                            after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                            targets = targets[after_area >= include_thresh * before_area]

                if scale:
                    p = np.random.uniform(0,1)
                    if p >= (1-scale[2]):
                        image, M, scale_factor = ssd_batch_generator._scale(self,image, scale[0], scale[1])
                        # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                        toplefts = np.array([targets[:,0], targets[:,2], np.ones(targets.shape[0])])
                        bottomrights = np.array([targets[:,1], targets[:,3], np.ones(targets.shape[0])])
                        new_toplefts = (np.dot(M, toplefts)).T
                        new_bottomrights = (np.dot(M, bottomrights)).T
                        targets[:,[0,2]] = new_toplefts.astype(np.int)
                        targets[:,[1,3]] = new_bottomrights.astype(np.int)
                        if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                            before_limiting = np.copy(targets)
                            x_coords = targets[:,[0,1]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            targets[:,[0,1]] = x_coords
                            y_coords = targets[:,[2,3]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            targets[:,[2,3]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                            after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                            targets = targets[after_area >= include_thresh * before_area]

                if crop:
                    image = image[crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]]

                    if limit_boxes: # Adjust boxes affected by cropping and remove those that will no longer be in the image
                        before_limiting = np.copy(targets)
                        if crop[0] > 0:
                            y_coords = targets[:,[2,3]]
                            y_coords[y_coords < crop[0]] = crop[0]
                            targets[:,[2,3]] = y_coords
                        if crop[1] > 0:
                            y_coords = targets[:,[2,3]]
                            y_coords[y_coords >= (img_height - crop[1])] = img_height - crop[1] - 1
                            targets[:,[2,3]] = y_coords
                        if crop[2] > 0:
                            x_coords = targets[:,[0,1]]
                            x_coords[x_coords < crop[2]] = crop[2]
                            targets[:,[0,1]] = x_coords
                        if crop[3] > 0:
                            x_coords = targets[:,[0,1]]
                            x_coords[x_coords >= (img_width - crop[3])] = img_width - crop[3] - 1
                            targets[:,[0,1]] = x_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                        after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                        targets = targets[after_area >= include_thresh * before_area]
                    # Now adjust the box coordinates for the new image size post cropping
                    if crop[0] > 0:
                        targets[:,[2,3]] -= crop[0]
                    if crop[2] > 0:
                        targets[:,[0,1]] -= crop[2]
                    img_height -= crop[0] - crop[1]
                    img_width -= crop[2] - crop[3]

                if resize:
                    image = cv2.resize(image, dsize=resize)
                    targets[:,[0,1]] = (targets[:,[0,1]] * (resize[0] / img_width)).astype(np.int)
                    targets[:,[2,3]] = (targets[:,[2,3]] * (resize[1] / img_height)).astype(np.int)

                if gray:
                    image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 3)

                if diagnostics:
                    processed_images.append(image)
                    processed_labels.append(targets)

                img = Image.fromarray(image.astype(np.uint8))
                img.save('{}{}'.format(dest_path, filename), 'JPEG', quality=90)
                del image
                del img
                gc.collect()

                # Transform the labels back to the original CSV file format:
                # One line per ground truth box, i.e. possibly multiple lines per image
                for target in targets:
                    target = list(target)
                    target = [filename] + target
                    targets_for_csv.append(target)

            with open('{}labels.csv'.format(dest_path), 'w', newline='') as csvfile:
                labelswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                labelswriter.writerow(['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
                labelswriter.writerows(targets_for_csv)

            if diagnostics:
                print("Image processing completed.")
                return np.array(processed_images), np.array(original_images), np.array(targets_for_csv), processed_labels
            else:
                print("Image processing completed.")    


class keras_layer_DecodeDetectionsFast:
    class DecodeDetectionsFast(Layer):

        def __init__(self,
                     confidence_thresh=0.01,
                     iou_threshold=0.45,
                     top_k=200,
                     nms_max_output_size=400,
                     coords='centroids',
                     normalize_coords=True,
                     img_height=None,
                     img_width=None,
                     **kwargs):
            if K.backend() != 'tensorflow':
                raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

            if normalize_coords and ((img_height is None) or (img_width is None)):
                raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

            if coords != 'centroids':
                raise ValueError("The DetectionOutput layer currently only supports the 'centroids' coordinate format.")

            # We need these members for the config.
            self.confidence_thresh = confidence_thresh
            self.iou_threshold = iou_threshold
            self.top_k = top_k
            self.normalize_coords = normalize_coords
            self.img_height = img_height
            self.img_width = img_width
            self.coords = coords
            self.nms_max_output_size = nms_max_output_size

            # We need these members for TensorFlow.
            self.tf_confidence_thresh = tf.constant(self.confidence_thresh, name='confidence_thresh')
            self.tf_iou_threshold = tf.constant(self.iou_threshold, name='iou_threshold')
            self.tf_top_k = tf.constant(self.top_k, name='top_k')
            self.tf_normalize_coords = tf.constant(self.normalize_coords, name='normalize_coords')
            self.tf_img_height = tf.constant(self.img_height, dtype=tf.float32, name='img_height')
            self.tf_img_width = tf.constant(self.img_width, dtype=tf.float32, name='img_width')
            self.tf_nms_max_output_size = tf.constant(self.nms_max_output_size, name='nms_max_output_size')

            super(keras_layer_DecodeDetectionsFast.DecodeDetectionsFast, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            super(keras_layer_DecodeDetectionsFast.DecodeDetectionsFast, self).build(input_shape)

        def call(self, y_pred, mask=None):
            class_ids = tf.expand_dims(tf.to_float(tf.argmax(y_pred[...,:-12], axis=-1)), axis=-1)
            # Extract the confidences of the maximal classes.
            confidences = tf.reduce_max(y_pred[...,:-12], axis=-1, keep_dims=True)

            # Convert anchor box offsets to image offsets.
            cx = y_pred[...,-12] * y_pred[...,-4] * y_pred[...,-6] + y_pred[...,-8] # cx = cx_pred * cx_variance * w_anchor + cx_anchor
            cy = y_pred[...,-11] * y_pred[...,-3] * y_pred[...,-5] + y_pred[...,-7] # cy = cy_pred * cy_variance * h_anchor + cy_anchor
            w = tf.exp(y_pred[...,-10] * y_pred[...,-2]) * y_pred[...,-6] # w = exp(w_pred * variance_w) * w_anchor
            h = tf.exp(y_pred[...,-9] * y_pred[...,-1]) * y_pred[...,-5] # h = exp(h_pred * variance_h) * h_anchor

            # Convert 'centroids' to 'corners'.
            xmin = cx - 0.5 * w
            ymin = cy - 0.5 * h
            xmax = cx + 0.5 * w
            ymax = cy + 0.5 * h

            # If the model predicts box coordinates relative to the image dimensions and they are supposed
            # to be converted back to absolute coordinates, do that.
            def normalized_coords():
                xmin1 = tf.expand_dims(xmin * self.tf_img_width, axis=-1)
                ymin1 = tf.expand_dims(ymin * self.tf_img_height, axis=-1)
                xmax1 = tf.expand_dims(xmax * self.tf_img_width, axis=-1)
                ymax1 = tf.expand_dims(ymax * self.tf_img_height, axis=-1)
                return xmin1, ymin1, xmax1, ymax1
            def non_normalized_coords():
                return tf.expand_dims(xmin, axis=-1), tf.expand_dims(ymin, axis=-1), tf.expand_dims(xmax, axis=-1), tf.expand_dims(ymax, axis=-1)

            xmin, ymin, xmax, ymax = tf.cond(self.tf_normalize_coords, normalized_coords, non_normalized_coords)

            # Concatenate the one-hot class confidences and the converted box coordinates to form the decoded predictions tensor.
            y_pred = tf.concat(values=[class_ids, confidences, xmin, ymin, xmax, ymax], axis=-1)

            #####################################################################################
            # 2. Perform confidence thresholding, non-maximum suppression, and top-k filtering.
            #####################################################################################

            batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
            n_boxes = tf.shape(y_pred)[1]
            n_classes = y_pred.shape[2] - 4
            class_indices = tf.range(1, n_classes)

            # Create a function that filters the predictions for the given batch item. Specifically, it performs:
            # - confidence thresholding
            # - non-maximum suppression (NMS)
            # - top-k filtering
            def filter_predictions(batch_item):

                # Keep only the non-background boxes.
                positive_boxes = tf.not_equal(batch_item[...,0], 0.0)
                predictions = tf.boolean_mask(tensor=batch_item,
                                              mask=positive_boxes)

                def perform_confidence_thresholding():
                    # Apply confidence thresholding.
                    threshold_met = predictions[:,1] > self.tf_confidence_thresh
                    return tf.boolean_mask(tensor=predictions,
                                           mask=threshold_met)
                def no_positive_boxes():
                    return tf.constant(value=0.0, shape=(1,6))

                # If there are any positive predictions, perform confidence thresholding.
                predictions_conf_thresh = tf.cond(tf.equal(tf.size(predictions), 0), no_positive_boxes, perform_confidence_thresholding)

                def perform_nms():
                    scores = predictions_conf_thresh[...,1]

                    # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                    xmin = tf.expand_dims(predictions_conf_thresh[...,-4], axis=-1)
                    ymin = tf.expand_dims(predictions_conf_thresh[...,-3], axis=-1)
                    xmax = tf.expand_dims(predictions_conf_thresh[...,-2], axis=-1)
                    ymax = tf.expand_dims(predictions_conf_thresh[...,-1], axis=-1)
                    boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                    maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                                  scores=scores,
                                                                  max_output_size=self.tf_nms_max_output_size,
                                                                  iou_threshold=self.iou_threshold,
                                                                  name='non_maximum_suppresion')
                    maxima = tf.gather(params=predictions_conf_thresh,
                                       indices=maxima_indices,
                                       axis=0)
                    return maxima
                def no_confident_predictions():
                    return tf.constant(value=0.0, shape=(1,6))

                # If any boxes made the threshold, perform NMS.
                predictions_nms = tf.cond(tf.equal(tf.size(predictions_conf_thresh), 0), no_confident_predictions, perform_nms)

                def top_k():
                    return tf.gather(params=predictions_nms,
                                     indices=tf.nn.top_k(predictions_nms[:, 1], k=self.tf_top_k, sorted=True).indices,
                                     axis=0)
                def pad_and_top_k():
                    padded_predictions = tf.pad(tensor=predictions_nms,
                                                paddings=[[0, self.tf_top_k - tf.shape(predictions_nms)[0]], [0, 0]],
                                                mode='CONSTANT',
                                                constant_values=0.0)
                    return tf.gather(params=padded_predictions,
                                     indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                     axis=0)

                top_k_boxes = tf.cond(tf.greater_equal(tf.shape(predictions_nms)[0], self.tf_top_k), top_k, pad_and_top_k)

                return top_k_boxes

            # Iterate `filter_predictions()` over all batch items.
            output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                      elems=y_pred,
                                      dtype=None,
                                      parallel_iterations=128,
                                      back_prop=False,
                                      swap_memory=False,
                                      infer_shape=True,
                                      name='loop_over_batch')

            return output_tensor

        def compute_output_shape(self, input_shape):
            batch_size, n_boxes, last_axis = input_shape
            return (batch_size, self.tf_top_k, 6) # Last axis: (class_ID, confidence, 4 box coordinates)

        def get_config(self):
            config = {
                'confidence_thresh': self.confidence_thresh,
                'iou_threshold': self.iou_threshold,
                'top_k': self.top_k,
                'nms_max_output_size': self.nms_max_output_size,
                'coords': self.coords,
                'normalize_coords': self.normalize_coords,
                'img_height': self.img_height,
                'img_width': self.img_width,
            }
            base_config = super(keras_layer_DecodeDetectionsFast.DecodeDetectionsFast, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class keras_layer_AnchorBoxes:
    class AnchorBoxes(Layer):

        def __init__(self,
                     img_height,
                     img_width,
                     this_scale,
                     next_scale,
                     aspect_ratios=[0.5, 1.0, 2.0],
                     two_boxes_for_ar1=True,
                     this_steps=None,
                     this_offsets=None,
                     limit_boxes=True,
                     variances=[1.0, 1.0, 1.0, 1.0],
                     coords='centroids',
                     normalize_coords=False,
                     **kwargs):
            
            if K.backend() != 'tensorflow':
                raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

            if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
                raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))

            if len(variances) != 4:
                raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
            variances = np.array(variances)
            if np.any(variances <= 0):
                raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

            self.img_height = img_height
            self.img_width = img_width
            self.this_scale = this_scale
            self.next_scale = next_scale
            self.aspect_ratios = aspect_ratios
            self.two_boxes_for_ar1 = two_boxes_for_ar1
            self.this_steps = this_steps
            self.this_offsets = this_offsets
            self.limit_boxes = limit_boxes
            self.variances = variances
            self.coords = coords
            self.normalize_coords = normalize_coords
            # Compute the number of boxes per cell
            if (1 in aspect_ratios) and two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios) + 1
            else:
                self.n_boxes = len(aspect_ratios)
            super(keras_layer_AnchorBoxes.AnchorBoxes, self).__init__(**kwargs)

        def build(self, input_shape):
            self.input_spec = [InputSpec(shape=input_shape)]
            super(keras_layer_AnchorBoxes.AnchorBoxes, self).build(input_shape)

        def call(self, x, mask=None):
            size = min(self.img_height, self.img_width)
            # Compute the box widths and and heights for all aspect ratios
            wh_list = []
            for ar in self.aspect_ratios:
                if (ar == 1):
                    # Compute the regular anchor box for aspect ratio 1.
                    box_height = box_width = self.this_scale * size
                    wh_list.append((box_width, box_height))
                    if self.two_boxes_for_ar1:
                        # Compute one slightly larger version using the geometric mean of this scale value and the next.
                        box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                        wh_list.append((box_width, box_height))
                else:
                    box_height = self.this_scale * size / np.sqrt(ar)
                    box_width = self.this_scale * size * np.sqrt(ar)
                    wh_list.append((box_width, box_height))
            wh_list = np.array(wh_list)

            # We need the shape of the input tensor
    #         print(K.image_data_format)
            if K.image_data_format() == 'channels_last':
                batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
            else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
                batch_size, feature_map_channels, feature_map_height, feature_map_width = x._keras_shape

            # Compute the grid of box center points. They are identical for all aspect ratios.

            # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
            if (self.this_steps is None):
                step_height = self.img_height / feature_map_height
                step_width = self.img_width / feature_map_width
            else:
                if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                    step_height = self.this_steps[0]
                    step_width = self.this_steps[1]
                elif isinstance(self.this_steps, (int, float)):
                    step_height = self.this_steps
                    step_width = self.this_steps
            # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
            if (self.this_offsets is None):
                offset_height = 0.5
                offset_width = 0.5
            else:
                if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                    offset_height = self.this_offsets[0]
                    offset_width = self.this_offsets[1]
                elif isinstance(self.this_offsets, (int, float)):
                    offset_height = self.this_offsets
                    offset_width = self.this_offsets
            # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
            cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
            cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
            cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

            # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
            # where the last dimension will contain `(cx, cy, w, h)`
            boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

            boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
            boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
            boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
            boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

            #ssd_box_encode_decode_utils = ssd_box_encode_decode_utils()
            # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
            boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='centroids2corners')

            # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
            if self.limit_boxes:
                x_coords = boxes_tensor[:,:,:,[0, 2]]
                x_coords[x_coords >= self.img_width] = self.img_width - 1
                x_coords[x_coords < 0] = 0
                boxes_tensor[:,:,:,[0, 2]] = x_coords
                y_coords = boxes_tensor[:,:,:,[1, 3]]
                y_coords[y_coords >= self.img_height] = self.img_height - 1
                y_coords[y_coords < 0] = 0
                boxes_tensor[:,:,:,[1, 3]] = y_coords

            # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
            if self.normalize_coords:
                boxes_tensor[:, :, :, [0, 2]] /= self.img_width
                boxes_tensor[:, :, :, [1, 3]] /= self.img_height

            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth.
            if self.coords == 'centroids':
                # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
                boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='corners2centroids')
            elif self.coords == 'minmax':
                # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
                boxes_tensor = ssd_box_encode_decode_utils.convert_coordinates(self,boxes_tensor, start_index=0, conversion='corners2minmax')

            # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
            # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
            variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
            variances_tensor += self.variances # Long live broadcasting
            # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
            boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

            # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
            boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
            boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

            return boxes_tensor

        def compute_output_shape(self, input_shape):
            if K.image_data_format() == 'channels_last':
                batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
            else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
                batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
            return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

        def get_config(self):
            config = {
                'img_height': self.img_height,
                'img_width': self.img_width,
                'this_scale': self.this_scale,
                'next_scale': self.next_scale,
                'aspect_ratios': list(self.aspect_ratios),
                'two_boxes_for_ar1': self.two_boxes_for_ar1,
                'limit_boxes': self.limit_boxes,
                'variances': list(self.variances),
                'coords': self.coords,
                'normalize_coords': self.normalize_coords
            }
            base_config = super(keras_layer_AnchorBoxes.AnchorBoxes, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))


class keras_ssd_loss:
    class SSDLoss(object):
        '''
        The SSD loss, see https://arxiv.org/abs/1512.02325.
        '''

        def __init__(self,
                     neg_pos_ratio=3,
                     n_neg_min=0,
                     alpha=1.0):
            
            self.neg_pos_ratio = neg_pos_ratio
            self.n_neg_min = n_neg_min
            self.alpha = alpha

        def smooth_L1_loss(self, y_true, y_pred):
            
            absolute_loss = tf.abs(y_true - y_pred)
            square_loss = 0.5 * (y_true - y_pred)**2
            l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
            return tf.reduce_sum(l1_loss, axis=-1)

        def log_loss(self, y_true, y_pred):
            
            # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
            y_pred = tf.maximum(y_pred, 1e-15)
            # Compute the log loss
            log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
            return log_loss

        def compute_loss(self, y_true, y_pred):
            
            self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
            self.n_neg_min = tf.constant(self.n_neg_min)
            self.alpha = tf.constant(self.alpha)

            batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
            n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

            # 1: Compute the losses for class and box predictions for every box

            classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
            localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

            # 2: Compute the classification losses for the positive and negative targets

            # Create masks for the positive and negative ground truth classes
            negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
            positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # Tensor of shape (batch_size, n_boxes)

            # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
            n_positive = tf.reduce_sum(positives)

            pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

            # Compute the classification loss for the negative default boxes (if there are any)

            # First, compute the classification loss for all negative boxes
            neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
            n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) 
            # Compute the number of negative examples we want to account for in the loss
            # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
            n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

            # In the unlikely case when either (1) there are no negative ground truth boxes at all
            # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
            def f1():
                return tf.zeros([batch_size])
            # Otherwise compute the negative loss
            def f2():
                
                neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
                # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
                values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False) # We don't need sorting
                # ...and with these indices we'll create a mask...
                negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
                negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
                # ...and use it to keep only those boxes and mask all other classification losses
                neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
                return neg_class_loss

            neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

            class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

            # 3: Compute the localization loss for the positive targets
            #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

            loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

            total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
            
            total_loss *= tf.to_float(batch_size)

            return total_loss

    class FocalLoss(object):
        '''
        The SSD loss, see https://arxiv.org/abs/1512.02325.
        '''

        def __init__(self,
                     neg_pos_ratio=3,
                     n_neg_min=0,
                     alpha=1.0):
            
            self.neg_pos_ratio = neg_pos_ratio
            self.n_neg_min = n_neg_min
            self.alpha = alpha

        def smooth_L1_loss(self, y_true, y_pred):
            
            absolute_loss = tf.abs(y_true - y_pred)
            square_loss = 0.5 * (y_true - y_pred)**2
            l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
            return tf.reduce_sum(l1_loss, axis=-1)

        def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
            
            y_pred = tf.maximum(y_pred, 1e-15)
            log_y_pred = tf.log(y_pred)
            focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
            focal_loss = tf.multiply(y_true, tf.multiply(focal_scale, log_y_pred))
            return -tf.reduce_sum(focal_loss, axis=-1)

        def compute_loss(self, y_true, y_pred):
            
            self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
            self.n_neg_min = tf.constant(self.n_neg_min)
            self.alpha = tf.constant(self.alpha)

            batch_size = tf.shape(y_pred)[0] # Output dtype: tf.int32
            n_boxes = tf.shape(y_pred)[1] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

            # 1: Compute the losses for class and box predictions for every box

            classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) # Output shape: (batch_size, n_boxes)
            localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8])) # Output shape: (batch_size, n_boxes)

            # 2: Compute the classification losses for the positive and negative targets

            # Create masks for the positive and negative ground truth classes
            negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
            positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1)) # Tensor of shape (batch_size, n_boxes)

            # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
            n_positive = tf.reduce_sum(positives)

            # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
            # (Keras loss functions must output one scalar loss value PER batch item, rather than just
            # one scalar for the entire batch, that's why we're not summing across all axes)
            pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

            # Compute the classification loss for the negative default boxes (if there are any)

            # First, compute the classification loss for all negative boxes
            neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
            n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`
            
            # Compute the number of negative examples we want to account for in the loss
            # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
            n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

            # In the unlikely case when either (1) there are no negative ground truth boxes at all
            # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
            def f1():
                return tf.zeros([batch_size])
            # Otherwise compute the negative loss
            def f2():
                
                # To do this, we reshape `neg_class_loss_all` to 1D...
                neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
                # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
                values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False) # We don't need sorting
                # ...and with these indices we'll create a mask...
                negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
                negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
                # ...and use it to keep only those boxes and mask all other classification losses
                neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
                return neg_class_loss

            neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

            class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

           
            loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

            # 4: Compute the total loss

            total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive) # In case `n_positive == 0`
            
            total_loss *= tf.to_float(batch_size)

            return total_loss

    class weightedSSDLoss(SSDLoss):
        def __init__(self,
                     neg_pos_ratio=3,
                     n_neg_min=0,
                     alpha=1.0,
                     weights=None):

            super(keras_ssd_loss.weightedSSDLoss, self).__init__(neg_pos_ratio,
                                                  n_neg_min,
                                                  alpha)
            self.weights = weights

        def log_loss(self, y_true, y_pred):
            
            weighted = tf.multiply(y_true, self.weights)
            y_pred = tf.maximum(y_pred, 1e-15)
            # Compute the log loss
            xent = tf.multiply(y_true, tf.log(y_pred))
            log_loss = -tf.reduce_sum(weighted * xent, axis=-1)
            return log_loss

    class weightedFocalLoss(FocalLoss):
        def __init__(self,
                     neg_pos_ratio=3,
                     n_neg_min=0,
                     alpha=1.0,
                     weights=None):

            super(keras_ssd_loss.weightedFocalLoss, self).__init__(neg_pos_ratio,
                                                  n_neg_min,
                                                  alpha)
            self.weights = weights

        def log_loss(self, y_true, y_pred, gamma=2, alpha=0.5):
            
            weighted = tf.multiply(y_true, self.weights)
            y_pred = tf.maximum(y_pred, 1e-15)
            log_y_pred = tf.log(y_pred)
            focal_scale = tf.multiply(tf.pow(tf.subtract(1.0, y_pred), gamma), alpha)
            focal_loss = tf.multiply(weighted, tf.multiply(focal_scale, log_y_pred))
            return -tf.reduce_sum(focal_loss, axis=-1)

class mobilenet_v1:
    #depthwise_conv2d = depthwise_conv2d()
    def mobilenet(input_tensor):

        if input_tensor is None:
            input_tensor = Input(shape=(300,300,3))


        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_padding')(input_tensor)
        x = Convolution2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv0")(x)

        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name = "conv0/bn")(x)

        x = Activation('relu')(x)



        x = DepthwiseConvolution2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False, name="conv1/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv1/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, name="conv1")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv1/bn")(x)
        x = Activation('relu')(x)

        print ("conv1 shape: ", x.shape)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv2_padding')(x)
        x = DepthwiseConvolution2D(64, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv2/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv2/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv2")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv2/bn")(x)
        x = Activation('relu')(x)

        

        x = DepthwiseConvolution2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv3/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv3/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv3")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv3/bn")(x)
        x = Activation('relu')(x)

        print ("conv3 shape: ", x.shape)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv3_padding')(x)
        x = DepthwiseConvolution2D(128, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv4/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv4/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv4")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv4/bn")(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv5/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv5/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv5")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv5/bn")(x)
        x = Activation('relu')(x)

        print ("conv5 shape: ", x.shape)

        
        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv4_padding')(x)
        x = DepthwiseConvolution2D(256, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv6/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv6/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv6")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv6/bn")(x)
        x = Activation('relu')(x)

        test = x
        
        for i in range(5):
            x = DepthwiseConvolution2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False,name=("conv" + str(7+i)+"/dw" ))(x)
            x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name=("conv" + str(7+i)+"/dw/bn" ))(x)
            x = Activation('relu')(x)
            x = Convolution2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False,name=("conv" + str(7+i)))(x)
            x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name=("conv" + str(7+i) +"/bn"))(x)
            x = Activation('relu')(x)

        # print ("conv11 shape: ", x.shape)
        conv11 = x


        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv5_padding')(x)
        x = DepthwiseConvolution2D(512, (3, 3), strides=(2, 2), padding='valid', use_bias=False,name="conv12/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv12/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv12")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv12/bn")(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False,name="conv13/dw")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv13/dw/bn")(x)
        x = Activation('relu')(x)
        x = Convolution2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False,name="conv13")(x)
        x = BatchNormalization( momentum=0.99, epsilon=0.00001 , name="conv13/bn")(x)
        x = Activation('relu')(x)

        conv13 = x

        # print ("conv13 shape: ", x.shape)


        # model = Model(inputs=input_tensor, outputs=x)

        return [conv11,conv13,test]

class ssd_mobilenet:
    def ssd_300(mode,
                image_size,
                n_classes,
                l2_regularization=0.0005,
                min_scale=None,
                max_scale=None,
                scales=None,
                aspect_ratios_global=None,
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=None,
                limit_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                coords='centroids',
                normalize_coords=False,
                subtract_mean=[123, 117, 104],
                divide_by_stddev=None,
                swap_channels=True,
                return_predictor_sizes=False):
        

        n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
        n_classes += 1  # Account for the background class.
        l2_reg = l2_regularization  # Make the internal name shorter.
        img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]


        if aspect_ratios_global is None and aspect_ratios_per_layer is None:
            raise ValueError(
                "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
        if aspect_ratios_per_layer:
            if len(aspect_ratios_per_layer) != n_predictor_layers:
                raise ValueError(
                    "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                        n_predictor_layers, len(aspect_ratios_per_layer)))

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
        if scales:
            if len(scales) != n_predictor_layers + 1:
                raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                    n_predictor_layers + 1, len(scales)))
        else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
            scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if (not (steps is None)) and (len(steps) != n_predictor_layers):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ############################################################################
        # Compute the anchor box parameters.
        ############################################################################

        # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
        if aspect_ratios_per_layer:
            aspect_ratios = aspect_ratios_per_layer
        else:
            aspect_ratios = [aspect_ratios_global] * n_predictor_layers

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        if aspect_ratios_per_layer:
            n_boxes = []
            for ar in aspect_ratios_per_layer:
                if (1 in ar) & two_boxes_for_ar1:
                    n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
                else:
                    n_boxes.append(len(ar))
        else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                n_boxes = len(aspect_ratios_global) + 1
            else:
                n_boxes = len(aspect_ratios_global)
            n_boxes = [n_boxes] * n_predictor_layers

        if steps is None:
            steps = [None] * n_predictor_layers
        if offsets is None:
            offsets = [None] * n_predictor_layers



        x = Input(shape=(img_height, img_width, img_channels))

        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        x1 = Lambda(lambda z: z, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
        if not (subtract_mean is None):
            x1 = Lambda(lambda z: z - np.array(subtract_mean), output_shape=(img_height, img_width, img_channels),
                        name='input_mean_normalization')(x1)
        if not (divide_by_stddev is None):
            x1 = Lambda(lambda z: z / np.array(divide_by_stddev), output_shape=(img_height, img_width, img_channels),
                        name='input_stddev_normalization')(x1)
        if swap_channels and (img_channels == 3):
            x1 = Lambda(lambda z: z[..., ::-1], output_shape=(img_height, img_width, img_channels),
                        name='input_channel_swap')(x1)

        #mobilenet_v1 = mobilenet_v1()
        conv4_3_norm , fc7 ,test= mobilenet_v1.mobilenet(input_tensor=x1)

        print ("conv11 shape: ", conv4_3_norm.shape)
        print ("conv13 shape: ", fc7.shape)



        conv6_1 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv14_1', use_bias=False)(fc7)
        conv6_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv14_1/bn')(conv6_1)
        conv6_1 = Activation('relu', name='relu_conv6_1')(conv6_1)

        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv14_2', use_bias=False)(conv6_1)
        conv6_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv14_2/bn')(conv6_2)
        conv6_2 = Activation('relu', name='relu_conv6_2')(conv6_2)

        print ('conv14 shape', conv6_2.shape)



        conv7_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv15_1',use_bias=False)(conv6_2)
        conv7_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv15_1/bn')(conv7_1)
        conv7_1 = Activation('relu', name='relu_conv7_1')(conv7_1)

        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv15_2',use_bias=False)(conv7_1)
        conv7_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv15_2/bn')(conv7_2)
        conv7_2 = Activation('relu', name='relu_conv7_2')(conv7_2)


        print ('conv15 shape', conv7_2.shape)

        conv8_1 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv16_1',use_bias=False)(conv7_2)
        conv8_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv16_1/bn')(conv8_1)
        conv8_1 = Activation('relu', name='relu_conv8_1')(conv8_1)
        conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(conv8_1)
        conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv16_2',use_bias=False)(conv8_1)
        conv8_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv16_2/bn')(conv8_2)
        conv8_2 = Activation('relu', name='relu_conv8_2')(conv8_2)

        print ('conv16 shape', conv8_2.shape)
        
        conv9_1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv17_1',use_bias=False)(conv8_2)
        conv9_1 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv17_1/bn')(conv9_1)
        conv9_1 = Activation('relu', name='relu_conv9_1')(conv9_1)
        conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(conv9_1)
        conv9_2 = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg), name='conv17_2',use_bias=False)(conv9_1)
        conv9_2 = BatchNormalization( momentum=0.99, epsilon=0.00001, name='conv17_2/bn')(conv9_2)
        conv9_2 = Activation('relu', name='relu_conv9_2')(conv9_2)

        print ('conv17 shape', conv9_2.shape)

        # Feed conv4_3 into the L2 normalization layer
        # conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3_norm)
       

        conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                        kernel_regularizer=l2(l2_reg), name='conv11_mbox_conf')(conv4_3_norm)
        fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv13_mbox_conf')(fc7)
        conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_conf')(conv6_2)
        conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_conf')(conv7_2)
        conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_conf')(conv8_2)
        conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (1,1), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_conf')(conv9_2)
        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                       kernel_regularizer=l2(l2_reg), name='conv11_mbox_loc')(conv4_3_norm)
        fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv13_mbox_loc')(fc7)
        conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv14_2_mbox_loc')(conv6_2)
        conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv15_2_mbox_loc')(conv7_2)
        conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv16_2_mbox_loc')(conv8_2)
        conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (1,1), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=l2(l2_reg), name='conv17_2_mbox_loc')(conv9_2)

        ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        #keras_layer_AnchorBoxes = keras_layer_AnchorBoxes()
        conv4_3_norm_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                                 aspect_ratios=aspect_ratios[0],
                                                 two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                                 this_offsets=offsets[0], limit_boxes=limit_boxes,
                                                 variances=variances, coords=coords, normalize_coords=normalize_coords,
                                                 name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
        fc7_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                        aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                        limit_boxes=limit_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='fc7_mbox_priorbox')(fc7_mbox_loc)
        conv6_2_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                            aspect_ratios=aspect_ratios[2],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                            this_offsets=offsets[2], limit_boxes=limit_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
        conv7_2_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                            aspect_ratios=aspect_ratios[3],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                            this_offsets=offsets[3], limit_boxes=limit_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
        conv8_2_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                            aspect_ratios=aspect_ratios[4],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                            this_offsets=offsets[4], limit_boxes=limit_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
        conv9_2_mbox_priorbox = keras_layer_AnchorBoxes.AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                            aspect_ratios=aspect_ratios[5],
                                            two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                            this_offsets=offsets[5], limit_boxes=limit_boxes,
                                            variances=variances, coords=coords, normalize_coords=normalize_coords,
                                            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

        ### Reshape

        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)
        fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
        conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
        conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
        conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
        conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
        fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
        conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
        conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
        conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
        conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
            conv4_3_norm_mbox_priorbox)
        fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
        conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
        conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
        conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
        conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

        ### Concatenate the predictions from the different layers

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                           fc7_mbox_conf_reshape,
                                                           conv6_2_mbox_conf_reshape,
                                                           conv7_2_mbox_conf_reshape,
                                                           conv8_2_mbox_conf_reshape,
                                                           conv9_2_mbox_conf_reshape])

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                         fc7_mbox_loc_reshape,
                                                         conv6_2_mbox_loc_reshape,
                                                         conv7_2_mbox_loc_reshape,
                                                         conv8_2_mbox_loc_reshape,
                                                         conv9_2_mbox_loc_reshape])

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                                   fc7_mbox_priorbox_reshape,
                                                                   conv6_2_mbox_priorbox_reshape,
                                                                   conv7_2_mbox_priorbox_reshape,
                                                                   conv8_2_mbox_priorbox_reshape,
                                                                   conv9_2_mbox_priorbox_reshape])

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

        model = Model(inputs=x, outputs=predictions)
        # return model
        #keras_layer_DecodeDetectionsFast = keras_layer_DecodeDetectionsFast()
        if mode == 'inference':
            print ('in inference mode')
            decoded_predictions = keras_layer_DecodeDetectionsFast.DecodeDetectionsFast(confidence_thresh=0.01,
                                                       iou_threshold=0.45,
                                                       top_k=100,
                                                       nms_max_output_size=100,
                                                       coords='centroids',
                                                       normalize_coords=normalize_coords,
                                                       img_height=img_height,
                                                       img_width=img_width,
                                                       name='decoded_predictions')(predictions)
            model = Model(inputs=x, outputs=decoded_predictions)
        else:
            print ('in training mode')

        return model

