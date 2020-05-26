# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:56:16 2020
@author: Leo
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import json
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical  
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from models import SegNet_model, UNet_model, ENet_model
from models import DeepLab_model, PSPNet_model, ERFNet_model
from models import FCN32_model, FCN16_model, FCN8_model
from models_pro import SegNet_model_CBAM, SegNet_model_NLB
from models_pro import UNet_model_SGate, UNet_model_CGate, UNet_model_CSGate

def preprocess_img(img):
    '''
    图像转数组，归一化
    '''
    img = image.img_to_array(img)
    img = 2/255.0 * img - 1
    return img # -1~1
def augmentation_img(img1,img2):
    '''
    图像增强（旋转/翻转/缩放/亮度调整/噪声）
    '''
    aug = np.random.randint(2,size=5)
    img_a1, img_a2 = np.copy(img1) , np.copy(img2)
    
    if aug[0]:# 旋转
        angle  = np.random.choice((1,2,-1,-2))
        img_a1 = np.rot90(img_a1,angle)
        img_a2 = np.rot90(img_a2,angle)
    if aug[1]:# 对称翻转 
        x_or_y = np.random.choice((0,1))
        img_a1 = np.flip(img_a1,x_or_y)
        img_a2 = np.flip(img_a2,x_or_y)
    if aug[2]:# 缩放
        zoom = np.random.uniform(0.8,1.2)
        img_a1 = image.random_zoom(img_a1,(zoom,zoom),0,1,2) 
        img_a2 = image.random_zoom(img_a2,(zoom,zoom),0,1,2) 
    if aug[3]:# 亮度
        img_a1 = image.random_brightness(img_a1, (0.5,1.5))
    if aug[4]:# 噪声
        img_a1 += np.random.uniform(0,8,(np.shape(img_a1)))
        img_a1 = np.clip(img_a1,0,255)
    # 转成 0-255 整数
    img_a1 = np.array(img_a1,'uint8')
    img_a2 = np.array(img_a2,'uint8')
    return img_a1, img_a2

def batch_generater(data_path, data_list, batch_size, shape, n_label, training=True):
    offset = 0
    while True:
        train_list = data_list[0]
        test_list = data_list[1]
        X = np.zeros((batch_size, *shape))
        Y = np.zeros((batch_size, shape[0]*shape[1], n_label))
        for i in range(batch_size):
            img_x_path = data_path[0] + '/' + train_list[i + offset]
            img_y_path = data_path[1] + '/' + test_list[i + offset]
            
            img_x = image.load_img(img_x_path, target_size = shape[0:2])
            img_y = image.load_img(img_y_path, target_size = shape[0:2])
            img_x = image.img_to_array(img_x)
            img_y = image.img_to_array(img_y)
            img_y = img_y[...,0:1] # label图三个通道是一样的，只留一个
            
            if training:
                img_x, img_y = augmentation_img(img_x, img_y) # 图像增强
            img_x = preprocess_img(img_x) # 归一化
            img_y = np.array(img_y).flatten() # 展平
            img_y = to_categorical(img_y, n_label) # one-hot
            
            X[i,...] = img_x
            Y[i,...] = img_y
            if i+offset >= len(train_list)-1:
                if training:
                    data_list = shuffle(data_list)
                offset = 0
        yield (X, Y)
        offset += batch_size

def update_training_log():
    json_path = model_savepath + 'model_training/' + model_savename + '_training_curve.json'
    log_new = h.history
    if os.path.exists(json_path):
        with open(json_path, "rb") as f:
            log = f.read()
            log = json.loads(log)        
        log_new['acc']      = log['acc'] + log_new['acc']
        log_new['loss']     = log['loss'] + log_new['loss']
        log_new['val_acc']  = log['val_acc'] + log_new['val_acc']
        log_new['val_loss'] = log['val_loss'] + log_new['val_loss']
    with open(json_path,'w') as f:
        json.dump(log_new,f)
    return log_new

def plot_curve():
    json_path = model_savepath + 'model_training/' + model_savename + '_training_curve.json'
    if os.path.exists(json_path):
        with open(json_path, "rb") as f:
            log = f.read()
            log = json.loads(log)  
    plt.figure(1)
    plt.plot(log['loss'])
    plt.plot(log['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('{}/{}_loss_{:.4f}.jpg'.format(model_savepath, model_savename, log['val_loss'][-1]),dpi=600)
    plt.figure(2)
    plt.plot(log['acc'])
    plt.plot(log['val_acc'])
    plt.title('model train vs validation acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('{}/{}_acc_{:.4f}.jpg'.format(model_savepath, model_savename, log['val_acc'][-1]),dpi=600)
    return None

if __name__ == '__main__':
    # 设置超参数
    input_shape = (256, 256, 3)
    batch_size = 6
    epoch = 20
    n_label = 4 + 1
    # 指定文件夹划分数据集
    x_path = '../data/remote_sensing_image/train/src'   # 值域范围 0-255
    y_path = '../data/remote_sensing_image/train/label' # 值域范围 0-255
    train_list = os.listdir(x_path)
    test_list  = os.listdir(y_path)
    X_train, X_test, y_train, y_test = train_test_split(train_list, test_list, test_size=0.2, random_state=42)
    # 搭建模型 + 加载权重
    model = UNet_model(input_shape, n_label)
    adam  = Adam(lr=0.001, epsilon=1e-08, decay=1e-6, amsgrad=True)
    sgd   = SGD(lr=0.001, momentum=0.9, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
    #model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    model.summary()
    
    # 模型结构与权重 加载与保存机制
    model_savepath = './model/'
    model_savename = 'UNet_model_base'
    with open(model_savepath+model_savename+'.json', 'w') as f:
        f.write(model.to_json())
    if os.listdir(model_savepath):
        try: 
            model.load_weights(model_savepath + model_savename + '_weights.h5')
            print('Model weights loaded!')
        except:
            print('Model weights load filed!')
    # 模型设置，设置ckpt、提前结束
    save_best = ModelCheckpoint(model_savepath + model_savename + '_weights.h5', monitor='val_acc', 
                                verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    h = model.fit_generator(generator=batch_generater((x_path,y_path),(X_train,y_train),batch_size,input_shape,n_label), 
                            steps_per_epoch = len(X_train)//batch_size, 
                            epochs=epoch, verbose=1, callbacks=[save_best, early_stop], 
                            validation_steps = len(X_test)//32,
                            validation_data=batch_generater((x_path,y_path),(X_test,y_test),2,input_shape,n_label,training=False))
    update_training_log()
    plot_curve()
    