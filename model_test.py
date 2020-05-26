# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:06:51 2020

@author: Leo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

def padding_img(img, target_size):
    '''
    构造一个图片，长宽为 256 的整数倍
    多出的部分用 0 填充
    输出：padding之后的图像
    '''
    w, h, _ = np.shape(img)
    pad_img_w = (w//target_size[0]+1) * target_size[0]
    pad_img_h = (h//target_size[1]+1) * target_size[1]
    pad_img = np.zeros((pad_img_w, pad_img_h, target_size[2]))
    pad_img[:w, :h, :] = img
    return pad_img

def devide_img(img, target_size):
    '''
    将 pad_img 划分为 target_size 的形状
    处理对象：3通道图像/单通道图像均可
    输出：子图集合
    '''
    h, w, _ = np.shape(img) # 输入图像的长宽信息（padding之后的）
    if h%target_size[0] !=0 or w%target_size[1] !=0:
        raise ValueError('Shape matching error!')
    n_row = h//target_size[0] # 行数：长//行
    n_column = w//target_size[1] # 列数：宽//列
    imgs = np.zeros((n_row*n_column, *target_size))
    for i in range(n_row):
        for j in range(n_column):
            imgs[i*n_column+j,...] = img[i*target_size[0]:(i+1)*target_size[0],j*target_size[1]:(j+1)*target_size[1],:]
    return imgs

def preprocess_img(filepath,img_id,target_size):
    '''
    加载图片并转为数组，归一化（-1,1）
    按照target_size做padding
    剪裁成target_size形状的小图
    输出：子图集合，原图长宽
    '''
    img = image.load_img(filepath+'/'+img_id)
    img = np.array(img)
    h, w, _ = np.shape(img)
    print('原图形状：',img.shape)
    img = 2/255.0 * img - 1
    img  = padding_img(img, target_size)
    print('大图形状：',img.shape)
    imgs =  devide_img(img, target_size)
    print('子图集合：',imgs.shape[0])
    return imgs, h, w

def merge_img(imgs, h, w, target_size):
    '''
    将所有子图拼接为一张大图（长宽等于原图）
    注意：输入的 w、h 为原图长宽
    '''
    big_img_h = (h//target_size[0]+1) * target_size[0]
    big_img_w = (w//target_size[1]+1) * target_size[1]
    big_img = np.zeros((big_img_h, big_img_w, imgs.shape[-1])) # 根据子图集中图片通道数判断形状
    n_row = big_img_h//target_size[0]
    n_column = big_img_w//target_size[1] 
    for i in range(n_row):
        for j in range(n_column):
            big_img[i*target_size[0]:(i+1)*target_size[0],j*target_size[1]:(j+1)*target_size[1],:] = imgs[i*n_column+j,...]
    
    img = big_img[:h,:w,...]
    return img

def model_predict(target_size, model, img_set, h, w):
    '''
    一次性处理一张图片的所有子图
    1.由模型预测类别
    2.独热编码转整型编码
    3.reshape 成 nx256x256
    4.按顺序拼接
    '''
    # result1 = model.predict_classes(img_set, verbose=1)
    result = model.predict(img_set, verbose=1)
    result = np.argmax(result,axis=2).astype(np.int8)
    result = np.reshape(result,(img_set.shape[0],target_size[0],target_size[1]))
    result = np.expand_dims(result,axis=3)
    result = merge_img(result, h, w, target_size)
    result = result / 4
    result = image.array_to_img(result)
    plt.figure()
    plt.matshow(result)
    return result

def load_model_from_json(model_path, model_name):
    with open(model_path+model_name+'.json', 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(model_path+model_name+'_weights.h5')
    return model

if __name__ == '__main__':
    filepath = 'F:/Leo/深度学习/CV项目/语义分割/天空之眼/data/remote_sensing_image/test'
    #filepath = input('请输入绝对路径 Enter')
    test_set = os.listdir(filepath)
    target_size = (256, 256, 3)
    
    # 加载模型权重
    model_path = './model/'
    model_name = 'UNet_model_base'
    try:
        #model = load_model(model_path + model_name)
        model = load_model_from_json(model_path, model_name)
        print('Model weights loaded!')
    except:
        raise FileNotFoundError('Model not found!')
    
    for i in range(len(test_set)):
        print('开始处理第',str(i+1),'张图片...')
        img_id = test_set[i]
        img_set, h, w = preprocess_img(filepath,img_id,target_size)
        result = model_predict(target_size, model, img_set, h, w)
        result.save('./result/'+str(i+1)+'.png')
        


