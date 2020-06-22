
import tensorflow as tf
import yolo_model.utils as utils
from config.config import cfg
import math
import numpy as np



# Batch Normalization -> BN
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


# CBM and CBL block
def conv_BN_Leaky_or_Mish(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)

    return conv


# Mish Activation
def mish(x):
    """
    Mish Activation Function.
    math:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
    


# Res_Unit blocks
def res_unit(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = conv_BN_Leaky_or_Mish(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = conv_BN_Leaky_or_Mish(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output


# upsampling
def upsampling(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')


#  DARKNET_53
def darknet53(input_data):

    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3,  3,  32))
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 32,  64), downsample=True)

    # block 1
    for i in range(1):
        input_data = res_unit(input_data,  64,  32, 64)

    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3,  64, 128), downsample=True)

    # block 2
    for i in range(2):
        input_data = res_unit(input_data, 128,  64, 128)

    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 128, 256), downsample=True)

    # block 3
    for i in range(8):
        input_data = res_unit(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 256, 512), downsample=True)

    # block 4
    for i in range(8):
        input_data = res_unit(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 512, 1024), downsample=True)
    # block 5
    for i in range(4):
        input_data = res_unit(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


#  CSP_DARKNET_53
#  returns: output layer -> layer_54, layer_85, layer_104
def csp_darknet53(input_data):

    # CBM block x2
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3,  3,  32), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")
    # CSP_Res block x1
    route = input_data
    route = conv_BN_Leaky_or_Mish(route, (1, 1, 64, 64), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = res_unit(input_data,  64,  32, 64, activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    # CBM block x2
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 128, 64), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    # CSP_Res block x2
    route = input_data
    route = conv_BN_Leaky_or_Mish(route, (1, 1, 128, 64), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = res_unit(input_data, 64,  64, 64, activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    # CBM block x2
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    # CSP_Res block x8
    route = input_data
    route = conv_BN_Leaky_or_Mish(route, (1, 1, 256, 128), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = res_unit(input_data, 128, 128, 128, activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    # CBM block x1
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 256, 256), activate_type="mish")
    # route_1 -> layer_54
    route_1 = input_data
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    # CSP_Res block x8
    route = input_data
    route = conv_BN_Leaky_or_Mish(route, (1, 1, 512, 256), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = res_unit(input_data, 256, 256, 256, activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 512, 512), activate_type="mish")
    # route_2 -> layer_85
    route_2 = input_data
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = conv_BN_Leaky_or_Mish(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = res_unit(input_data, 512, 512, 512, activate_type="mish")
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    # layer_104
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 1024, 1024), activate_type="mish")

    # route_1 -> layer_54
    # route_2 -> layer_85
    # input_data -> layer_104
    return route_1, route_2, input_data


# SPP
def SPP(input_data):
    """ CBL block x3 + SPP + CBL block x3 """   
    
    # CBL block x3
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 1024, 512))
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 512, 1024))
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 1024, 512))

    # SPP -> (5x5), (9x9), (13x13)
    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), 
                            tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1), 
                            tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    # CBL block x3
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 2048, 512))
    input_data = conv_BN_Leaky_or_Mish(input_data, (3, 3, 512, 1024))
    input_data = conv_BN_Leaky_or_Mish(input_data, (1, 1, 1024, 512))

    return input_data


# modified PAN + head part
def PAN_modified(layer_116, layer_85, layer_54, NUM_CLASS):

    route = layer_116 
    conv = conv_BN_Leaky_or_Mish(layer_116, (1, 1, 512, 256))
    conv = upsampling(conv)
    route_2 = conv_BN_Leaky_or_Mish(layer_85, (1, 1, 512, 256))    # route_2 -> layer_85 -> 121
    conv = tf.concat([route_2, conv], axis=-1)

    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 256, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 256, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))     # layer_126
    
    route_2 = conv          # route_2 = layer_126
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 256, 128))
    conv = upsampling(conv)

    route_1 = conv_BN_Leaky_or_Mish(layer_54, (1, 1, 256, 128))     # layer_54 -> 130
    conv = tf.concat([route_1, conv], axis=-1)

    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 256, 128))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 128, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 256, 128))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 128, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 256, 128))    # layer_136

    route_1 = conv      # layrer_136
    # head part: + [CBL_block + 1x1Conv] -> (76x76x255) for small objects
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 128, 256))
    conv_sbbox = conv_BN_Leaky_or_Mish(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)    # layrer_138

    conv = conv_BN_Leaky_or_Mish(route_1, (3, 3, 128, 256), downsample=True)    # 136 -> 137
    conv = tf.concat([conv, route_2], axis=-1)    # layrer_126

    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 256, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 256, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 256))    # layer_147
    
    route_2 = conv    # layrer_147
    # head part: + [CBL_block + 1x1Conv] -> (38x38x255) for middle objects    
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 256, 512))
    conv_mbbox = conv_BN_Leaky_or_Mish(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = conv_BN_Leaky_or_Mish(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)    # route = layrer_116

    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 1024, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 512, 1024))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 1024, 512))
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 512, 1024))
    conv = conv_BN_Leaky_or_Mish(conv, (1, 1, 1024, 512))
    # head part: + [CBL_block + 1x1Conv] -> (19x19x255) for big objects
    conv = conv_BN_Leaky_or_Mish(conv, (3, 3, 512, 1024))
    conv_lbbox = conv_BN_Leaky_or_Mish(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]




#  YOLOv4 -> SPP + PAN + head
#  Returns: 
#   conv_sbbox(76x76x255) -> laery_138 
#   conv_mbbox(38x38x255) -> layer_149
#   conv_lbbox(19x19x255) -> layer_159
#  255 = (bboxes predcted by every cell) * [(x,y,w,h,confidence) + num_class]= 3 * (5 + 80)
def YOLOv4(input_layer, NUM_CLASS):

    # csp_darknet53
    layer_54, layer_85, layer_104 = csp_darknet53(input_layer)
    # SPP
    layer_116 = SPP(layer_104)
    # modified PAN  + head
    return PAN_modified(layer_116, layer_85, layer_54, NUM_CLASS)




