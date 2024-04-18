import struct
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.backends.backend_pdf import PdfPages
import time
import torch
from torch import nn
class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out+=residual
        out=self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,num_class):
        #inplane=当前的fm的通道数
        self.inplane=64
        super(ResNet, self).__init__()

        #参数
        self.block=block
        self.layers=layers

        #stem的网络层
        self.conv1=nn.Conv2d(3,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2)

        #后续的网络
        self.avgpool=nn.AvgPool2d(7)
        self.fc=nn.Linear(512*block.extention,num_class)

    def forward(self,x):
        #stem部分：conv+bn+maxpool
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block部分
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)

        #分类
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)

        return out

    def make_layer(self,block,plane,block_num,stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list=[]
        #先计算要不要加downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(plane*block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)

# 训练集标签和图像文件路径
train_labels_file = './MNIST/train-labels.idx1-ubyte'
train_images_file = './MNIST/train-images.idx3-ubyte'

# 测试集标签和图像文件路径
test_labels_file = './MNIST/t10k-labels.idx1-ubyte'
test_images_file = './MNIST/t10k-images.idx3-ubyte'
def simple_scatterplot(ax, X, selected,color):
    ax.scatter(X[:,0], X[:,1], s=8)
    ax.scatter(X[selected,0], X[selected,1], s=40, c=color)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_facecolor((241/255, 241/255, 246/255))
    for spine in ax.spines.values(): spine.set_edgecolor((159/255, 158/255, 180/255))
def read_idx_file(file_path):
    with open(file_path, 'rb') as f:
        # 读取头部信息
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]

        # 读取标签数据
        if magic_number == 2049:  # 判断是否为标签文件
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        # 读取图像数据
        elif magic_number == 2051:  # 判断是否为图像文件
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            pixels = np.frombuffer(f.read(), dtype=np.uint8)
            images = pixels.reshape(num_items, num_rows, num_cols)
            return images
        else:
            raise ValueError("未知的文件类型或数据格式错误！")
class SubsetSelection(object):
    def __init__(self, Y,reg):
        self.Y=Y
        self.reg = reg
    def ADMM_SparseModeling_bycvxpy(self, max_iter, p=np.inf,outlier=False,select_num=10,flag=True):
        Y=self.Y.T
        # 定义优化变量C
        print(Y.shape)
        [M, N] = np.shape(Y)
        C = cp.Variable((N, N))

        # 定义两个约束
        constraint1 = cp.sum(C,axis=0) == 1

        # 将约束放入 constraints 列表
        constraints = [constraint1]

        # 构建目标函数
        obj = cp.Minimize(self.reg * cp.norm(C, p) + 0.5 * cp.norm(Y - Y @ C, "fro"))

        # 创建问题实例
        problem = cp.Problem(obj, constraints)

        # 使用ADMM求解器
        print(max_iter)
        problem.solve(abstol=1e-5,reltol=1e-5, feastol=1e-5)
        # 获取最小值
        result = problem.value   
        [M,N]=np.array(C.value).shape
        norms = np.zeros(N)  # 存储列向量的q范数
        indices = np.arange(N)  # 列向量的索引
        top_k_indices = []  # 选择的列向量的索引
        for j in range(N):
            norms[j] = np.linalg.norm(C.value[:, j], ord=p)
            if norms[j]>0.3:
                top_k_indices.append(j)
            #print("范数，根据范数从小到大选择",norms[j])
        # 根据列向量的q范数排序
        sorted_indices = np.argsort(norms)[::-1]
        # 选择排序后的前k个向量的索引
        top_k_indices1 = sorted_indices[:int(len(norms)*0.4)]
        ret = {
             'C': C.value, 'selected': top_k_indices,'selected1':top_k_indices1,'target':result
        }
        return ret,C.value



# 读取训练集标签和图像数据
train_labels = read_idx_file(train_labels_file)
train_images = read_idx_file(train_images_file)

# 读取测试集标签和图像数据
test_labels = read_idx_file(test_labels_file)
test_images = read_idx_file(test_images_file)
print("训练集标签：", train_labels.shape)
X = torch.tensor(np.stack([train_images] * 3, axis=1), dtype=torch.float32)
print(X.shape)
#resnet50提取feature
resnet=ResNet(Bottleneck,[3,4,6,3],1000)
print(X.shape)
start=time.time()
X=resnet(X)
end=time.time()
print(end-start)
print(X.shape)
SS = SubsetSelection(X, 100)
result4, C = SS.ADMM_SparseModeling_bycvxpy(2e4,np.inf, False, 4,False)
result4=result4['selected1']
fig, axes = plt.subplots(1, 1, figsize=(6.5,6.5))

# Plot result4 on the second subplot
simple_scatterplot(axes, X, result4, 'brown')
axes.set_title('result')
axes.set_xticks([])
axes.set_yticks([])
# Set the title
fig.suptitle(f'Sparse Modeling (reg={100}, mu={0.001})')
# Save the figure to the PDF file
plt.show()