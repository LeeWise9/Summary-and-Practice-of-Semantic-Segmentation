# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:05:50 2020

@author: 李睿
"""

# utils
# 集合不同的模型结构
'''
SegNet
UNet
FCN
ENet
DeepLab
PSPNet
ERFNet
'''

import keras
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, PReLU, LeakyReLU
from keras.layers import UpSampling2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Reshape, Permute, Activation, Lambda
from keras.layers import concatenate, Add, Subtract, Average
from keras.layers.core import Dropout, SpatialDropout2D
from keras.models import Model
from keras.backend import tf as ktf

###############################################################################
# 一些公共的卷积计算模块
def conv2d_layers(x, n_kernal, size_kernal=3):
    '''普通卷积模块：卷积 + BN'''
    x = Conv2D(n_kernal, (size_kernal,size_kernal), strides=(1, 1), 
               padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv2d_transpose(x, n_kernal, t_stride=2, size_kernal=2):
    '''转置卷积模块'''
    x = Conv2DTranspose(n_kernal, kernel_size=size_kernal, strides=t_stride, 
                        padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv2d_dilation(x, n_kernal, d_rate, size_kernal=3):
    '''空洞卷积模块'''
    x = Conv2D(n_kernal, kernel_size=size_kernal, strides=(1,1), padding='same', 
               dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv2d_depthwise(x, n_kernal, size_kernal=3):
    '''深度可分离卷积模块'''
    x = Conv2D(n_kernal,kernel_size=(1, 1),padding='same',kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConv2D(kernel_size=size_kernal, padding='same',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(int(n_kernal*2),kernel_size=(1, 1),padding='same',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def OutPuts(x, n_label, input_shape):
    '''将卷积层reshape成对应的输出形状'''
    x = Conv2D(n_label,(1, 1),strides=(1, 1), padding='same')(x)
    x = Reshape((input_shape[0]*input_shape[1], n_label))(x)
    x = Activation('softmax')(x)
    return x

# Encoder
def Encoder(x):
    '''公共的下采样部分'''
    block = conv2d_layers(x, 32)
    block = conv2d_layers(block, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block)
    
    block = conv2d_layers(pool1, 64)
    block = conv2d_layers(block, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block)
    
    block = conv2d_layers(pool2, 128)
    block = conv2d_layers(block, 128)
    block = conv2d_layers(block, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block)
    
    block = conv2d_layers(pool3, 256)
    block = conv2d_layers(block, 256)
    block = conv2d_layers(block, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(block)
    
    block = conv2d_layers(pool4, 256)
    block = conv2d_layers(block, 256)
    block = conv2d_layers(block, 256)
    pool5 = MaxPooling2D(pool_size=(2, 2))(block)
    
    return [x, pool1, pool2, pool3, pool4, pool5]





###############################################################################
# SegNet
'''
Encoder + (skip) + Decoder
上采样：upsampling + convs
'''
def SegNet_model(input_shape, n_label, skip=True):
    input_img = Input(shape=input_shape)
    # encoder
    encoder = Encoder(input_img)
    pool1, pool2, pool3, pool4, pool5 = encoder[1], encoder[2], encoder[3], encoder[4], encoder[5]
    # 1/2    1/4    1/8   1/16   1/32
    # decoder
    block = UpSampling2D(size=(2, 2))(pool5)
    block = conv2d_layers(block,256)
    block = conv2d_layers(block,256)
    block = conv2d_layers(block,256)
    if skip:
        block = Add()([block, pool4])
    # 1/16
    block = UpSampling2D(size=(2, 2))(block)
    block = conv2d_layers(block,128)
    block = conv2d_layers(block,128)
    block = conv2d_layers(block,128)
    if skip:
        block = Add()([block, pool3])
    # 1/8
    block = UpSampling2D(size=(2, 2))(block)
    block = conv2d_layers(block,64)
    block = conv2d_layers(block,64)
    block = conv2d_layers(block,64)
    if skip:
        block = Add()([block, pool2])
    # 1/4
    block = UpSampling2D(size=(2, 2))(block)
    block = conv2d_layers(block,32)
    block = conv2d_layers(block,32)
    if skip:
        block = Add()([block, pool1])
    # 1/2
    block = UpSampling2D(size=(2, 2))(block)
    block = conv2d_layers(block,32)
    block = conv2d_layers(block,32)
    # 1
    outputs = OutPuts(block, n_label, input_shape)

    model = Model(inputs=input_img, outputs=outputs)
    return model

###############################################################################
# UNet
'''
U型网络
Encoder + skip + Decoder
上采样：转置卷积
'''
def UNet_model(input_shape, n_label):
    input_img = Input(input_shape)
    
    conv1 = conv2d_layers(input_img, 32)
    conv1 = conv2d_layers(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = conv2d_layers(pool1, 64)
    conv2 = conv2d_layers(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 64
    conv3 = conv2d_layers(pool2, 128)
    conv3 = conv2d_layers(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 32
    conv4 = conv2d_layers(pool3, 256)
    conv4 = conv2d_layers(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 16
    conv5 = conv2d_layers(pool4, 512)
    conv5 = conv2d_layers(conv5, 512)
    
    # 16
    conv5 = conv2d_transpose(conv5, 256, t_stride=2, size_kernal=2)
    upsp6 = concatenate([conv5, conv4], axis=3)
    conv6 = conv2d_layers(upsp6, 256)
    conv6 = conv2d_layers(conv6, 256)
    # 32
    conv6 = conv2d_transpose(conv6, 128, t_stride=2, size_kernal=2)
    upsp7 = concatenate([conv6, conv3], axis=3)
    conv7 = conv2d_layers(upsp7, 128)
    conv7 = conv2d_layers(conv7, 128)
    # 64
    conv7 = conv2d_transpose(conv7, 64, t_stride=2, size_kernal=2)
    upsp8 = concatenate([conv7, conv2], axis=3)
    conv8 = conv2d_layers(upsp8, 64)
    conv8 = conv2d_layers(conv8, 64)
    # 128
    conv8 = conv2d_transpose(conv8, 32, t_stride=2, size_kernal=2)
    upsp9 = concatenate([conv8, conv1], axis=3)
    conv9 = conv2d_layers(upsp9, 32)
    conv9 = conv2d_layers(conv9, 32)
    # 256
    outputs = OutPuts(conv9, n_label, input_shape)
    
    model = Model(inputs=input_img, outputs=outputs)
    return model


###############################################################################
# FCN family
'''
FCN 32/16/8 
Encoder + 转置卷积(32/16/8)
'''
def FCN_conv(x, n_label):
    '''FCN中用卷积替代全连接的部分'''
    '''计算量很大，酌情删减'''
    x = conv2d_layers(x, 512, 8)
    x = Dropout(0.5)(x)
    x = conv2d_layers(x, 512, 1)
    x = Dropout(0.5)(x)
    x = conv2d_layers(x, n_label, 1)
    return x

# FCN32
def FCN32_model(input_shape, n_label):
    input_img = Input(input_shape)
    encoder = Encoder(input_img)[5]
    x = FCN_conv(encoder, n_label)
    x = conv2d_layers(x, n_label, 1)
    x = conv2d_transpose(x, n_label, t_stride=32, size_kernal=64)
    
    outputs = OutPuts(x, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model

# FCN16
def FCN16_model(input_shape, n_label):
    input_img = Input(input_shape)
    encoder = Encoder(input_img)
    x1, x2 = encoder[4], encoder[5] # 1/16 & 1/32
    
    x1 = FCN_conv(x1, n_label)
    x2 = FCN_conv(x2, n_label)
    x2 = conv2d_transpose(x2, n_label, t_stride=2, size_kernal=4)
    x  = Add()([x1, x2])
    x  = conv2d_transpose(x, n_label, t_stride=16, size_kernal=32)
    
    outputs = OutPuts(x, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model

# FCN8
def FCN8_model(input_shape, n_label):
    input_img = Input(input_shape)
    encoder = Encoder(input_img)
    x1, x2, x3 = encoder[3], encoder[4], encoder[5] # 1/8 & 1/16 & 1/32
    
    x1 = FCN_conv(x1, n_label)
    x2 = FCN_conv(x2, n_label)
    x3 = FCN_conv(x3, n_label)
    x2 = conv2d_transpose(x2, n_label, t_stride=2, size_kernal=4)
    x3 = conv2d_transpose(x3, n_label, t_stride=4, size_kernal=8)
    x = Add()([x1, x2, x3])
    x = conv2d_transpose(x, n_label, t_stride=8, size_kernal=16)
    
    outputs = OutPuts(x, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model


###############################################################################
# ENet
'''
大量使用可分离卷积，空洞卷积，跳跃连接
激活函数使用 PReLU
Dropout函数使用 SpatialDropout2D
BN层 momentum=0.1

ENet_initial_block：初始化模块
ENet_bottleneck：下采样模块
ENet_up_bottleneck：上采样模块
'''
def ENet_initial_block(x):
    x1 = Conv2D(13, (3,3), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    # x1 = BatchNormalization()(x1)
    x2 = MaxPooling2D(pool_size=(2, 2))(x)
    x  = concatenate([x1, x2], axis=3)
    x  = BatchNormalization(momentum=0.1)(x)
    x  = PReLU(shared_axes=[1, 2])(x)
    return x # 1/2

def ENet_bottleneck(x, output_filters, dp_rate=0.1, downsample=False, asymmetric=0, dilated=0):
    '''
    1.降采样时：maxpooling+padding + 2x2+3x3+1x1
    2.不降采样：input + 1x1+3x3+1x1
    兼容空间、深度可分离卷积和空洞卷积
    '''
    stride = 2 if downsample else 1
    input_filters = output_filters // 4
    # 1x1
    x1 = Conv2D(input_filters, kernel_size=stride, strides=stride, 
                padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x1 = BatchNormalization(momentum=0.1)(x1)
    x1 = PReLU(shared_axes=[1, 2])(x1)
    # 3x3
    if not asymmetric and not dilated:
        x1 = Conv2D(input_filters, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
    elif asymmetric:
        x1 = Conv2D(input_filters, (1, asymmetric), padding='same', kernel_initializer='he_normal')(x1)
        x1 = Conv2D(input_filters, (asymmetric, 1), padding='same', kernel_initializer='he_normal')(x1)
    elif dilated:
        x1 = Conv2D(input_filters, (3, 3), dilation_rate=(dilated, dilated), padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization(momentum=0.1)(x1)
    x1 = PReLU(shared_axes=[1, 2])(x1)
    # 1x1
    x1 = Conv2D(output_filters, (1,1), padding='same', kernel_initializer='he_normal', use_bias=False)(x1)
    x1 = BatchNormalization(momentum=0.1)(x1)
    x1 = SpatialDropout2D(dp_rate)(x1)  # 按行 / 按列 Dropout
    
    x2 = MaxPooling2D()(x)
    #x2 = ZeroPadding2D(padding=(1, 1))
    if downsample:
        x  = concatenate([x1, x2], axis=3)
    else:
        x = concatenate([x1, x], axis=3)
    x  = PReLU(shared_axes=[1, 2])(x)
    return x

def ENet_up_bottleneck(x, output_filters, upsample=False):
    input_filters = output_filters // 4
    inp = x
    # 1x1
    x = Conv2D(input_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x) # x1
    # 3x3
    if upsample: # x2
        x = conv2d_transpose(x, input_filters, t_stride=2, size_kernal=2)
    else: # x1
        x = Conv2D(input_filters, (3, 3), padding='same', use_bias=True)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    # 1x1
    x = Conv2D(output_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    
    if upsample:
        inp = Conv2D(output_filters, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(inp)
        inp = BatchNormalization(momentum=0.1)(inp)
        inp = UpSampling2D(size=(2, 2))(inp)
    x = Add()([x, inp])
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    return x

def ENet_model(input_shape, n_label):
    input_img = Input(input_shape)
    # initial 1/2
    init = ENet_initial_block(input_img)     
    # bottleneck1.0 1/4     
    enet = ENet_bottleneck(init, 64, 0.01, downsample=True) 
    # 4x bottleneck1.x
    for _ in range(4):
        enet = ENet_bottleneck(enet, 64, 0.01)
    # bottleneck2.0 1/8
    enet = ENet_bottleneck(enet, 128, downsample=True) 
    # bottleneck2.x  3.x
    for _ in range(2):
        enet = ENet_bottleneck(enet, 128)               # bottleneck 2.1
        enet = ENet_bottleneck(enet, 128, dilated=2)    # bottleneck 2.2
        enet = ENet_bottleneck(enet, 128, asymmetric=5) # bottleneck 2.3
        enet = ENet_bottleneck(enet, 128, dilated=4)    # bottleneck 2.4
        enet = ENet_bottleneck(enet, 128)               # bottleneck 2.5
        enet = ENet_bottleneck(enet, 128, dilated=8)    # bottleneck 2.6
        enet = ENet_bottleneck(enet, 128, asymmetric=5) # bottleneck 2.7
        enet = ENet_bottleneck(enet, 128, dilated=16)   # bottleneck 2.8

    enet = ENet_up_bottleneck(enet, 64, upsample=True)  # bottleneck 4.0
    enet = ENet_up_bottleneck(enet, 64)                 # bottleneck 4.1
    enet = ENet_up_bottleneck(enet, 64)                 # bottleneck 4.2
    enet = ENet_up_bottleneck(enet, 16, upsample=True)  # bottleneck 5.0
    enet = ENet_up_bottleneck(enet, 16)                 # bottleneck 5.1
    
    enet = conv2d_transpose(enet, n_label, t_stride=2, size_kernal=2)
    outputs = OutPuts(enet, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model



###############################################################################
# DeepLab_v3
'''
使用不同膨胀率的空洞卷积提取不同尺度的信息
骨干网络使用深度可分离卷积模块减少参数
骨干网络使用了Xception模型
ASPP
'''
def DeepLab_model(input_shape, n_label):
    input_img = Input(input_shape)
    # 256
    conv1 = conv2d_depthwise(input_img, 32)
    conv1 = conv2d_depthwise(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128
    conv2 = conv2d_depthwise(pool1, 64)
    conv2 = conv2d_depthwise(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 64
    conv3 = conv2d_depthwise(pool2, 128)
    conv3 = conv2d_depthwise(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 32
    conv4 = conv2d_depthwise(pool3, 256)
    conv4 = conv2d_depthwise(conv4, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 16
    # spatial pyramid pooling
    c0 = conv2d_layers(pool4, 32, 1)
    c1 = conv2d_dilation(pool4, 32, 6)   # d_rate=6
    c1 = conv2d_layers(c1, 32)
    c2 = conv2d_dilation(pool4, 32, 12)  # d_rate=12
    c2 = conv2d_layers(c2, 32)
    c3 = conv2d_dilation(pool4, 32, 18)  # d_rate=18
    c3 = conv2d_layers(c3, 32)
    conv5 = concatenate([c0, c1, c2, c3])
    # 16
    conv6 = conv2d_depthwise(conv5, 128)
    conv6 = conv2d_depthwise(conv6, 128)
    conv6 = UpSampling2D(size=(4, 4))(conv6)
    # 64
    conv7 = concatenate([conv6, pool2])
    conv7 = conv2d_depthwise(conv7, 64)
    conv7 = conv2d_depthwise(conv7, 64)
    conv7 = UpSampling2D(size=(4, 4))(conv7)
    # 256
    conv8 = conv2d_layers(conv7, 32)
    outputs = OutPuts(conv8, n_label, input_shape)
    
    model = Model(inputs=input_img, outputs=outputs)
    return model


###############################################################################
# PSPNet
# 原文使用resnet作为骨干网络
'''
金字塔池化，提取不同尺度信息
插值缩放在测试阶段没跑通
'''
def resbolck(x, n_kernal):
    x1 = Conv2D(n_kernal, (1,1), strides=(1, 1), activation='relu', 
                padding='same', kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal, (3,3), strides=(1, 1), activation='relu',
                padding='same', kernel_initializer='he_normal')(x1)
    x1 = Conv2D(n_kernal, (1,1), strides=(1, 1), activation='relu',
                padding='same', kernel_initializer='he_normal')(x1)
    x = concatenate([x1, x], axis=3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Interp(x, shape):
    '''双线性插值，配合Lambda使用'''
    new_height, new_width = shape
    x = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return x

def interp_block(x, level, output_shape):
    # 默认 input_shape 为 256x256
    kernel_strides_map = {1: 32, 2: 16, 3: 10, 6: 5}
    kernel  = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    
    x = AveragePooling2D(kernel, strides=strides)(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(Interp, arguments={'shape': output_shape})(x)
    return x

def PSPblock(x, input_shape):
    '''金字塔池化模块'''
    '''输入：256//8，输出：256//8'''
    output_shape  = [input_shape[0]//8, input_shape[1]//8]
    interp_block1 = interp_block(x, 1, output_shape)
    interp_block2 = interp_block(x, 2, output_shape)
    interp_block3 = interp_block(x, 3, output_shape)
    interp_block6 = interp_block(x, 6, output_shape)
    x = concatenate([x, interp_block6, interp_block3, interp_block2, interp_block1], axis=3)
    return x

def PSPNet_model(input_shape, n_label):
    input_img = Input(input_shape)
    # 初始层
    x = conv2d_layers(input_img, 16, size_kernal=3)
    # resbolck
    for _ in range(3):
        for _ in range(5):
            x = resbolck(x, 32)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = conv2d_layers(x, 128, size_kernal=1)
    # 1/8
    # PSP 金字塔池化
    x = PSPblock(x, input_shape)
    x = conv2d_layers(x, 512)
    x = conv2d_layers(x, n_label, size_kernal=1)
    # 缩放到原图片大小
    x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)
    outputs = OutPuts(x, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model


###############################################################################
# ERFNet 
# -- Efficient Residual Factorized ConvNet
'''
核心操作是 residual connections 和 factorized convolutions(空间可分离卷积)
'''
def ERF_non_bottleneck(x, d_rate):
    n_kernal = int(x.shape[-1])
    x1 = Conv2D(n_kernal, (3,3), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal, (3,3), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x1)
    x  = Add()([x, x1])
    x  = BatchNormalization()(x)
    x  = Activation('relu')(x)
    return x

def ERF_bottleneck(x, d_rate):
    n_kernal = int(x.shape[-1])
    x1 = Conv2D(n_kernal//4, (1,1), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal//4, (3,3), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x1)
    x1 = Conv2D(n_kernal, (1,1), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x1)
    x  = Add()([x, x1])
    x  = BatchNormalization()(x)
    x  = Activation('relu')(x)
    return x

def ERF_non_bottleneck_1d(x, d_rate=1):
    n_kernal = int(x.shape[-1])
    x1 = Conv2D(n_kernal, (3,1), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal, (1,3), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal, (3,1), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x1 = Conv2D(n_kernal, (1,3), strides=(1, 1), activation='relu', padding='same', 
                dilation_rate=d_rate, kernel_initializer='he_normal')(x)
    x  = Add()([x, x1])
    x  = BatchNormalization()(x)
    x  = Activation('relu')(x)
    return x

def ERF_downsampler_block(x, n_kernal):
    w0 = n_kernal
    wi = int(x.shape[-1])
    x1 = MaxPooling2D(pool_size=(2, 2))(x)
    x2 = Conv2D(w0-wi, (3,3), strides=(2, 2), activation='relu',
                padding='same', kernel_initializer='he_normal')(x)
    x  = concatenate([x1, x2], axis=3)
    x  = BatchNormalization()(x)
    x  = Activation('relu')(x)
    return x

def ERFNet_model(input_shape, n_label):
    input_img = Input(input_shape)
    
    x = ERF_downsampler_block(input_img, n_kernal=16) # 1/2  c=16
    x = ERF_downsampler_block(x, n_kernal=64)         # 1/4  c=64
    for _ in range(5):
        x = ERF_non_bottleneck_1d(x, d_rate=64)       # 1/4  c=64
    x = ERF_downsampler_block(x, n_kernal=128)        # 1/8  c=128
    for _ in range(2):
        x = ERF_non_bottleneck_1d(x, d_rate=2)        # 1/8  c=128
        x = ERF_non_bottleneck_1d(x, d_rate=4)
        x = ERF_non_bottleneck_1d(x, d_rate=8)
        x = ERF_non_bottleneck_1d(x, d_rate=16)
    x = conv2d_transpose(x, 64, 2, 2)                 # 1/4  c=64
    for _ in range(2):
        x = ERF_non_bottleneck_1d(x)                  # 1/4  c=64
    x = conv2d_transpose(x, 16, 2, 2)                 # 1/2  c=16
    for _ in range(2):
        x = ERF_non_bottleneck_1d(x)                  # 1/2  c=16
    x = conv2d_transpose(x, n_label, 2, 2)            # 1/1  c=n_label
    
    outputs = OutPuts(x, n_label, input_shape)
    model = Model(inputs=input_img, outputs=outputs)
    return model



###############################################################################
# 

































if __name__ == '__main__':
    input_shape = (256,256,3)
    n_label = 5

    model = ERFNet_model(input_shape, n_label)
    model.summary()
    print(model.output_shape)



























