import argparse
import os
from datetime import datetime
from time import sleep

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
# preprocess the data set
from keras.datasets import mnist
from sklearn.decomposition import PCA
from torchvision import transforms

import deepcore.datasets as datasets
import deepcore.methods as methods
import deepcore.nets as nets
from utils import *


def ExtractFeature(train_images,type):
    if type=="original":
        train_images_flat = train_images.reshape(train_images.shape[0], -1)
        return train_images_flat
    elif type=="PCA":
        # 1. 将图像展平为一维向量
        train_images_flat = train_images.reshape(train_images.shape[0], -1)

        # 2. 对图像进行标准化，将像素值缩放到[0, 1]范围
        train_images_normalized = train_images_flat / 255.0
        # 3. 可选：进行降维
        # 假设您希望将图像降维到k维
        k = 100  # 设置目标降维维度
        pca = PCA(n_components=k)
        train_images_reduced = pca.fit_transform(train_images_normalized)
        return train_images_reduced
    elif type=="pretrained":
        # 创建一个数据预处理变换，将图像调整为ResNet所需的输入尺寸
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为RGB格式
            transforms.ToTensor(),
        ])
        # 对每张图像进行预处理，得到适合ResNet的输入
        processed_images = torch.stack([preprocess(img) for img in train_images])
        model_ft = models.resnet18(pretrained=True)
        model_ft.fc_backup = model_ft.fc
        model_ft.fc = nn.Sequential()
        model_ft.eval()
        # 将图像输入ResNet模型，得到输出
        outputs = model_ft(processed_images)
        return outputs.detach().numpy()
    

img_rows, img_cols = 28, 28
local_file_path = './data/mnist.npz'  # 将此路径替换为你保存 mnist.npz 文件的路径

# # 加载本地数据集
# with np.load(local_file_path, allow_pickle=True) as f:
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     # 假设 x_test 是原始的图像数据，形状为 (n_samples, 28, 28)
#     n_samples = x_test.shape[0]
#     flattened_x_test=ExtractFeature(x_test,"pretrained")
#     x_test_scaled = scaler.fit_transform(flattened_x_test)
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)
#     x_test_pca = pca.fit_transform(x_test_scaled)  # 应用PCA到标准化后的数据上
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# x_train = x_train.astype('float32')
# x_train /= 255
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# x_test = x_test.astype('float32')
# x_test /= 255
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
from keras.models import load_model

model = load_model('./resnet_cifar10.h5')
parser = argparse.ArgumentParser(description='Parameter Processing')
args=parser.parse_args()
print(methods.__dict__.keys())
method = methods.__dict__["CrossEntropySampling"](x_test, args, 0.3,4,np.squeeze(y_test),model)
print("out function")
indices=method.select()["indices"]
print("len:",len(x_test),"indices:",len(indices))
x_test_pca = ExtractFeature(x_test,"PCA")
# 提取采样数据点的坐标
sampled_points = x_test_pca[indices]
# 提取非采样数据点的坐标
other_points = np.delete(x_test_pca, indices, axis=0)
print(x_test_pca,sampled_points,other_points)
# 绘制所有数据点
plt.figure(figsize=(8, 6))
plt.scatter(other_points[:, 0], other_points[:, 1], color='b', label='Other Points')
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color='r', label='Sampled Points')
# 添加图例和标签
plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of x_test')
# 显示图像
plt.show()    
