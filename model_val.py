# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:27:48 2020

@author: 李睿
"""

# 模型评估
import os
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from model_test import load_model_from_json
from model_train import preprocess_img, augmentation_img, batch_generater


if __name__ == '__main__':
    # 设置超参数
    input_shape = (256, 256, 3)
    batch_size = 30
    epoch = 20
    n_label = 4 + 1
    # 指定文件夹划分数据集
    x_path = '../data/remote_sensing_image/train/src/'
    y_path = '../data/remote_sensing_image/train/label/'
    train_list = os.listdir(x_path)
    test_list  = os.listdir(y_path)
    X_train, X_test, y_train, y_test = train_test_split(train_list, test_list, test_size=0.2, random_state=42)
    
    # 加载模型权重
    model_path = './model/'
    model_name = 'UNet_model_base'
    try:
        #model = load_model(model_path + model_name)
        model = load_model_from_json(model_path, model_name)
        print('Model weights loaded!')
    except:
        raise FileNotFoundError('Model not found!')
        
    adam = Adam(lr=0.0001, epsilon=1e-08, decay=1e-6, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 

    acc, loss = model.evaluate_generator(
            generator=batch_generater(
                    (x_path,y_path),(X_test,y_test), batch_size, input_shape, n_label, False), 
                    steps=len(X_test)//batch_size, verbose=1)
    
    info = '{} --acc: {:.5f} --loss: {:.5f}'.format(model_name, acc, loss)
    save_file = './model/model_val/{}.txt'.format(model_name)
    print(info)
    with open(save_file,'w') as f:
        f.write(info)
        f.write('\n')















