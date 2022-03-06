# Caffe学习

## 一、Caffe介绍

### 1. 基本背景

​	2013年贾杨清博士发布在Github上，核心语言C++，支持Python和matlab接口。

### 2. 运行环境

​	既可以在CPU上运行，也可以在GPU上运行。

### 3. 应用领域

​	主要应用与计算机视觉领域，如图像识别、目标识别、人脸识别、图像风格转换等。

## 二、Caffe在windows下的安装编译

### 1. 环境准备

​	操作系统：64位win8或者64位win10。

​	编译环境：Visual Studio 2013 Ultimate版本。

### 2. 安装流程

​	① 下载windows分支

​		https://github.com/BVLC/caffe/

​	② 解压缩包，并把caffe-windows\windows目录下的CommonSettings.props.example文件改名为CommonSettings.props，然后打开CommonSettings.props文件，修改：

​	==CpuOnlyBuild -> True==

​	==UseCudNN -> False==

​	③打开Caffe.sln，设置libcaffe属性为debug版本，然后设置属性->配置属性->C/C++->常规->将警告是为错误设置为否，然后生成解决方案，等待依赖包下载以及编译，在主目录中出现Build文件夹。

## 三、快速上手MNIST数据集分类

### 1. 下载数据集

http://yann.lecun.com/exdb/mnist/

### 2. 转换数据格式

​	原始数据集为二进制文件，需要转换为LEVELDB或者LMDB。

​	使用生成的convert_mnist_data.exe可执行文件进行转换。

​	① 创建convert_train_ladb.bat

​	② 编写批处理文件（也可以在命令行中执行）

​		命令 + 空格 + 参数 + 空格 + 参数 + 空格 +参数

​		执行数据转换程序 -> 命令

​		传入训练图片 -> 参数

​		传入训练图片的标签 -> 参数

​		转换后的数据存放在此目录 -> 参数		

### 3. 修改数据路径

​	修改网络模型描述文件caffe-window\examples\mnist\lenet_train_test.protxt

### 4. 修改超参数文件

​	caffe-windows\examples\mnist\lenet_solver.protxt

​	修改网络路径、结果文件存放目录、求解模式

### 5. 开始训练模型

​	创建训练批处理文件，使用caffe.exe可执行文件进行训练

​	可执行文件路径 + 空格 + train（训练模式）+ 空格 + 超参数文件路径

​	xxx\Build\x64\Debug\caffe.exe train -solver=xxx/examples/mnist/lenet_solver.protxt

### 6. 等待模型训练好，准备要测试的图片

​	bmp文件

### 7. 生成均值文件

​	bat文件夹下创建mnist_mean.bat

​	xxx\Build\x64\Debug\compute_image_mean.exe xxx\examples\mnist\lmdb\train_lmdb ^ 	xxx\examples\mnist\mean_file\mean.binaryproto

​	可执行文件路径 + 空格 + 训练数据 + 空格 + 均值文件保存路径

​	作用：提高图像识别准确率

### 8. 准备标签

​	创建标签文件：examples\mnist\label\label.txt

​	标签文件内容：

​		0

​		1

​		…

​		9

### 9. 测试分类效果

​	新建批处理文件：mnist_classfication.bat

​	% 分类可执行程序

​	% 网络结构

​	% 训练好的模型

​	% 均值文件

​	% 标签

​	% 要分类的图片

​	xxx\Build\x64\Debug\classfication.exe xxx\examples\mnist\lenet.protxt ^

​	xxx\examples\mnist\lenet_iter_10000.caffemodel xxx\examples\mnist\mean.binaryproto ^

​	xxx\examples\mnist\label\label.txt xxx\examples\mnist\MNIST_data\0-9\5.bmp

​	pause

​	输出5个分类可能性，置信度从高到低排序。

## 四、Caffe文件详解

### 1. MNIST[数据集](http://yann.lecun.com/exdb/mnist/)

   下载下来的数据集被分成两个部分：60000张图片的训练数据集和10000张图片的测试数据集。

   每一张图片包含28*28个像素，图片里的每个像素都是8位的，也就是说每一个像素的强度介于0-255之间。

### 2. 下载原始数据集并转换

​	下载的原始数据为二进制文件，需要转换为LEVELDB或者LMDB格式。

​	LMDB(Lightning Memory Database Manager) - 闪电般的内存映射型数据数据库管理器，在caffe中的主要作用是进行数据管理，将各类类型的原始数据（比如JPEG图片，二进制数据）统一转换为Key-Value存储，以便于Caffe的DataLayer获取这些数据。而LEVELDB是Google开发的一种数据存储方式，在Caffe早期的版本中用得比较多，现在LMDB会用得比较多。

> 特点：数据格式的统一、数据IO读取快

### ==3. 修改网络模型描述文件==

​	xxx\examples\mnist\lenet_train_test.prototxt

```
name: "LeNet"			# 网络的名字是“LeNet”
layer {					# 定义一个层
  name: "mnist"			# 层的名字是“mnist”
  type: "Data" 			# 层的类型“Data”，表明数据来源于LevelDB或者LMDB。
  						# 另外数据的来源还可能是来自内存、HDF5、图片等（需修改）。
  top: "data" 			# 输出data
  top: "label"			# 输出label
  include {
    phase: TRAIN 		# 该层只在TRAIN时才加载（有效）
  }
  transform_param {		# 数据的预处理
    scale: 0.00390625 	# 每个像素乘以该值做归一化（1/255 = 0.00390625）
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb" # 前面生成的训练数据集
    batch_size: 64 		# 每一批训练集大小，处理64张图片
    backend: LMDB 		# 数据格式LMDB
  }
}
layer {					# 定义一个层
  name: "mnist"			# 层的名字“mnist”
  type: "Data" 			# 层的类型“Data”，表明数据来源于LevelDB或者LMDB。
  top: "data"			# 输出data
  top: "label"			# 输出label
  include {
    phase: TEST 		# 该层只在TEST时才加载（有效）
  }
  transform_param {		# 数据的预处理
    scale: 0.00390625	# 每个像素乘以该值做归一化（1/255 = 0.00390625）
    					# 将输入的数据0-255归一化到0-1之间
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb" # 前面生成的测试数据集
    batch_size: 100		# 每个批次出来100张图片
    backend: LMDB		# 数据格式LMDB
  }
}
layer {					# 定义一个层
  name: "conv1"			# 层的名字“conv1”
  type: "Convolution" 	# 卷积层，层的类型“Convolution”
  bottom: "data"		# 输入data
  top: "conv1"			# 输出conv1
  param {				# 这个是权值的学习率
    lr_mult: 1 			# 学习率系数。最终的学习率是这个学习率系数lr_mult乘以
    					# solver.prototxt里面的base_lr
  }
  param {				# 这个是偏置的学习率
    lr_mult: 2 			# 学习率系数。最终的学习率是这个学习率系数lr_mult乘以
    					# solver.prototxt里面的base_lr
  }
  convolution_param {
    num_output: 20 		# 输出多少个特征图（对应卷积核数量），卷积核的个数为20。
    kernel_size: 5 		# 卷积核大小5*5。果卷积核长和宽不等，
    					# 要kernel_h和kernel_w分别设置
    stride: 1 			# 步长为1，也可以用stride_h和stride_w来设置
    weight_filler {		# 权值初始化
      type: "xavier" 	# 使用“Xavier”算法，也可设置为“gaussian”
    }
    bias_filler {		# 偏置初始化
      type: "constant"  # 一般设置为“constant”，取值为0
    }
  }
}
layer {					# 定义一个层
  name: "pool1"			# 层的名字“pool1”
  type: "Pooling" 		# 层的类型“Pooling”
  bottom: "conv1" 		# 输入conv1
  top: "pool1"			# 输出pool1
  pooling_param {
    pool: MAX 			# 池化方法.常用的方法有MAX，AVE或STOCHASTIC
    kernel_size: 2		# 池化核的大小2*2.如果池化核长和宽不等
    					# 则需要用kernel_w和kernel_w分别设置
    stride: 2			# 池化的步长。也可以用stride_w和stride_w来设置
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50		# 卷积核的个数为50，或者表示输出特征平面的个数为50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {					# 定义一个层
  name: "ip1"			# 层的名字是“ip1”
  type: "InnerProduct" 	# 全链接层，层的类型“InnerProduct”
  bottom: "pool2"		# 输入pool2
  top: "ip1"			# 输出ip1
  param {
    lr_mult: 1 			# weights学习率
  }
  param {
    lr_mult: 2  		# bias学习率
  }
  inner_product_param {
    num_output: 500		# 500个神经元
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {					# 定义一个层
  name: "relu1"			# 层的名字是“relu1”
  type: "ReLU" 			# relu层，层的类型“ReLU”
  bottom: "ip1"			# 输入ip1
  top: "ip1"			# 输出ip1
}
layer {					# 定义一个层
  name: "ip2"			# 层的名字是“ip2”
  type: "InnerProduct"	# 层的类型“InnerProduct”，全连接层
  bottom: "ip1"			# 输入“ip1”
  top: "ip2"			# 输出“ip2”
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10		# 10个输出，代表10个分类
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}						
layer {					# 定义一个层
  name: "accuracy"		# 层的名字“accuracy”
  type: "Accuracy" 		# 层的类型“Accuracy”，用来判断准确率
  bottom: "ip2"			# 层的输入ip2
  bottom: "label"		# 层的输入label
  top: "accuracy"		# 层的输出accuracy
  include {
    phase: TEST			# 该层只在TEST测试的时候有效
  }
}
layer {					# 定义一个层
  name: "loss" 			# 层的名字“loss”
  type: "SoftmaxWithLoss" # 输出损失，层的类型“SoftmaxWithLoss”
  bottom: "ip2"			# 层的输入ip2
  bottom: "label"		# 层的输入label
  top: "loss"			# 层的输出loss
}
```

### 4. 修改超参数文件

​	xxxx\examples\mnist\lenet_solver.prototxt

```
# 网络模型描述文件
# 也可以用train_net和test_net来对训练模型和测试模型分别设定
# train_net: "xxxxxxxxxx"
# test_net: "xxxxxxxxxx"
net: "xxx/examples/mnist/lenet_train_test.prototxt"
# 测试迭代的次数，这个参数要跟test_layer结合起来考虑，
# 在test_layer中一个batch是100，而总共的测试图片是10000张
# 所以这个参数就是10000/100=100
test_iter: 100
# 每训练500次进行一次测试，每次测试都是以全部的测试集来测试
test_interval: 500
# 学习率
base_lr: 0.01
# 动力
momentum: 0.9
# type:SGD #优化算法的选择。这一行可以省略，因为默认值就是SGD，Caffe中一共有6中优化算法可以选择
# Stochastic Gradient Descent (type: "SGD"), 在Caffe中SGD其实应该是Momentum
# AdaDelta (type: "AdaDelta"),
# Adaptive Gradient (type: "AdaGrad"),
# Adam (type: "Adam"),
# Nesterov’s Accelerated Gradient (type: "Nesterov")
# RMSprop (type: "RMSProp")
# 权重衰减项，其实也就是正则化项。作用是防止过拟合
weight_decay: 0.0005
# 学习率调整策略
# 如果设置为inv,还需要设置一个power, 返回base_lr * (1 + gamma * iter) ^ (- power)，
# 其中iter表示当前的迭代次数
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# 每训练100次屏幕上显示一次，如果设置为0则不显示
display: 100
# 最大迭代次数
max_iter: 2000
# 快照。可以把训练的model和solver的状态进行保存。每迭代5000次保存一次，如果设置为0则不保存
snapshot: 5000
snapshot_prefix: "xxx/examples/mnist/models"
# 选择运行模式
solver_mode: GPU
```

### 5. 开始训练模型

### 6. 等待模型训练好，准备要测试的图片

### 7. 生成均值文件

​		图片减去均值后，再进行训练和测试，会提高速度和精度。因此，一般在各种图像识别的模型中都会有这个操作，实际上就是计算所有训练的平均值，计算出来后，保存为一个均值文件，在以后的测试中，就可以直接使用这个均值来相减，而不需要对测试图片重新计算。

### 8. 准备标签

### 9. 测试分类效果

## 五、各种优化器介绍

### 1.  Optimizer

​			tf.train.GradientDescentOptimizer

​			tf.train.AdadeltaOptimizer

​			tf.train.AdagradOptimizer

​			tf.train.AdagradDAOptimizer

​			tf.train.MomentumOptimizer

​			tf.train.AdamOptimizer

​			tf.train.FtrlOptimizer

​			tf.train.ProximalGradientDescentOptimizer

​			tf.train.ProximalAdagradOptimizer

​			tf.train.RMSPropOptimizerOptimizer

### 2.  各种优化器对比

​			① 标准梯度下降法：

​					标准梯度下降法先计算所有样本汇总误差，然后根据总误差来更新权值

​			② 随机梯度下降法：

​					随机梯度下降法随机抽取一个样本来计算误差，然后更新权值

​			③ 批量梯度下降法：

​			批量梯度下降法算是一种折中的方案，从总样本中选取一个批次（比如一共有10000个样本，随机选取100个样本作为一个batch），然后计算这个batch的总误差，根据总误差来更新权值。

### 3.  参数

​		
$$
W：要训练的参数
$$

$$
J(W)：代价函数
$$

$$
\nabla_w J(W) : 代价函数的梯度
$$

$$
η： 学习率
$$

​		①SGD
$$
W=W-η·\nabla_w J(W;x^{(i)};y^{(i)})
$$
​		② Momentum
$$
γ：动力，通常设置为0.9\\
v_t = γv_{t-1}+η\nabla_wJ(W)\\
W = W - v_t
$$
当前权值的改变会受到上一次权值改变的影响，类似于小球向下滚动的时候带上了惯性，这样可以加快小球的向下速度。

​		③NAG(Nesterov accelerated gradient)
$$
v_t = γv_{t-1}+η\nabla_wJ(W-γv_{t-1})\\
W = W - v_t
$$
​		NAG在tf中跟Momentum合并在同一个函数tf.train.MomentumOptimize中，可以通过参数配置启用。

​		④Adagrad

​		
$$
i:代表第i个分类\\
t:代表出现次数\\
\epsilon：的作用是避免分母为0，取值一般为1e^{-8}\\
η:取值一般为0.01\\
g_{t,i}=\nabla_wJ(W_i)\\
W_{t+i} = W_t - \frac{η}{\sqrt{\sum_{t'=1}^{t}{(g_{t',i})^2+\epsilon}}}⊙g(t)
$$
他是基于SGD的一种算法，他的核心思想是对比较常见的数据给予它比较小的学习率去调整参数，对于比较罕见的数据给予他比较大的学习率去调整参数。

​		⑤RMSprop

​			RMS（Root Mean Square）是均方根的缩写。

​		⑥ Adadelta

​			使用Adadelta我们甚至不需要设置一个默认学习率，在Adadelta不需要使用学习率也可以达到一个非常好的效果。

## 六、Caffe的Python接口安装及模型可视化

### 1. Release版本

​		① libcaffe文件右键-》属性

​				-》配置改为Release

​				-》C/C++-》常规-》将警告视为错误改为否

​		② 窗口中将Debug改为Release

​		③ 重新生成解决方案

### 2. Windows下Anaconda2和Anaconda3共存

​		① 先安装anaconda3，安装在F:\Anaconda3

​		② 然后再安装anaconda2，需要安装在F:\Anaconda3\py2（）

​		③ 安装完成之后，在命令提示符里面直接输入python会启动Python3。

​			如果想使用python2的话先执行activate py2，然后执行python就可以使用python2了。

​			取消激活使用deactivate。

### 3. windows下编译pycaffe

​		① pip install protobuf

​		② pip install pydot

​		③ 安装Graphiz:

​			http://www.graphiz.org/Download_windows.php

​		④ 把Graphiz安装文件的bin目录加入系统环境

​		⑤ 修改CommonSettings.props文件

​			<PythonSupport>true</PythonSupport>

​			<PythonDir>F:\Anaconda3\envs\py2\\</PythonDir>

​		⑥ 编译

​		⑦ 把Build\x64\Release\pycaffe\caffe复制到Anaconda2的Lib\site-packages目录下

​		⑧ 把Build\x64\Release\pycaffe\caffe目录下所以文件复制到caffe-windows\python\caffe目录下

​		⑨ 测试import caffe

### 4. 使用绘制网络结构

​		① 使用draw_net.py

​			python执行draw_net.py

​			第一个参数： --rankdir TB		#TB表示TOP到BOTTOM，或者使用LR，从左到右

​			第二个参数：网络结构描述文件的路径

​			第三个参数：网络结构的输出路径 

![20190425113239611](20190425113239611.png)

​		② 使用在线画图软件（需要翻墙）

​			http://ethereon.github.io/netscope/#/editor

## 七、Caffe特征可视化以及学习曲线可视化

### 1. 特征图可视化

```python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import caffe


#网络结构描述文件
deploy_file = "F:/deep_learning/Caffe/caffe-windows/examples/mnist/lenet.prototxt"
#模型文件
model_file  = "F:/deep_learning/Caffe/caffe-windows/examples/mnist/models_iter_10000.caffemodel"
#测试图片
test_data   = "F:/deep_learning/Caffe/caffe-windows/examples/mnist/MNIST_data/0-9/8.bmp"
#特征图路径
feature_map_path = "F:/deep_learning/Caffe/caffe-windows/examples/mnist/draw_data/"

#编写一个函数，用于显示各层的参数,padsize用于设置图片间隔空隙,padval用于调整亮度 
def show_data(data, name, padsize=1, padval=0):
    
    #归一化。归一化到0-1之间。
    data -= data.min()
    data /= data.max()
    
    #根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))     #20开根号得到4点多，ceil()得到5.
    
    # 对于conv1，data.shape->(20,24,24)
    # padding=（（前面填补0个，后面填补n**2-data.shape[0]）
    #（前面填补0个，后面填补padsize个），（前面填补0个，后面填补padsize个）
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize))
    #常数值填充，填充0
    data = np.pad(data, padding, mode='constant', constant_values=padval)
    
    # 对于conv1，padding后data.shape->(25,25,25)。
    # 原来是20，现在是25;原来的图片是24*24，现在是25*25
    # 对于conv1，将(25,25,25)reshape->(5,5,25,25)再transpose->(5,25,5,25)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3))
    
    # 再将（n, W, n, H）变换成(n*w, n*H)
    data = data.reshape((n * data.shape[1], n * data.shape[3]))
    

    image_path = os.path.join(feature_map_path,name)#特征图路径
    plt.set_cmap('gray')#设置为灰度图
    plt.imsave(image_path,data)#保存生成的图片
    plt.axis('off')#不显示坐标
    
    print name
    #显示图片
    img=Image.open(image_path)
    plt.imshow(img)
    plt.show()


#----------------------------数据预处理---------------------------------
#初始化caffe.调用.Net()函数。出入三个参数。
net = caffe.Net(deploy_file, #网络结构描述文件 
                model_file,  #训练好的模型
                caffe.TEST)  #使用测试模式

#输出网络每一层的参数。
print [(k, v[0].data.shape) for k, v in net.params.items()]

#定义了数据的预处理。一般就这样定义。
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 

# python读取的图片文件格式为H×W×K(高度，宽度，通道数)，需转化为K×H×W（通道数，高度，宽度）。caffe中需要这样转换。
transformer.set_transpose('data', (2, 0, 1))

# python中将图片存储为[0-1]
# 如果模型输入用的是0~255的原始格式，则需要做以下转换。
python读入的数据是0-1之间的，所以不需要转换。（liupc:存疑）
# transformer.set_raw_scale('data', 255)

# caffe中图片是BGR格式，而原始格式是RGB，所以要转化。这里是黑白图片，所以不需要转换。
#transformer.set_channel_swap('data', (2, 1, 0))


#----------------------------数据运算---------------------------------
#读取图片
#参数color: True(default)是彩色图，False是灰度图
img = caffe.io.load_image(test_data,color=False)

# 数据输入、预处理
net.blobs['data'].data[...] = transformer.preprocess('data', img)

# 将输入图片格式转化为合适格式（与deploy文件相同）
net.blobs['data'].reshape(1, 1, 28, 28)

# 前向迭代，即分类。保存输出
out = net.forward()

# 输出结果为各个可能分类的概率分布
print "Prob:"
print out['prob']

#最可能分类
predict = out['prob'].argmax()
print "Result:" + str(predict)

#----------------------------输出特征图---------------------------------
#第一个卷积层输出的特征图
#20指第一个卷积层有20个卷积核，输出20个特征平面。得到了20个24*24的特征平面。特征平面的图像保存为conv1.jpg
feature = net.blobs['conv1'].data
show_data(feature.reshape(20,24,24),'conv1.jpg')        
#第一个池化层输出的特征图
#池化后还是20个特征平面。
feature = net.blobs['pool1'].data
show_data(feature.reshape(20,12,12),'pool1.jpg')
#第二个卷积层输出的特征图
feature = net.blobs['conv2'].data
show_data(feature.reshape(50,8,8),'conv2.jpg')
#第二个池化层输出的特征图
feature = net.blobs['pool2'].data
show_data(feature.reshape(50,4,4),'pool2.jpg')
```



### 2. 训练loss和accuracy可视化

​		① 把caffe-windows\tools\extra目录下的plot_training_log.py.example

​			拷贝一份改名为plot_traing_log.py

​		② 获得log文件

​				xxx\Build\x64\Debug\caffe.exe train -solver=xxx/examples/mnist/lenet_solver.protxt ^ 				2>>mnist.log

​		③ python执行plot_training_log.py

​			第一个参数：0-7

​				训练或者测试的accuracy，loss等数据

​			第二个参数：图片存放的位置

​			第三个参数：log文件

​				python xxx\tools\extra\plot_training_log.py 0 test.png xxx\mnist.log

​		画图出现错误可以参考下面文章：

​		http://blog.csdn.net/sunshine_in_monn/article/details/53541573

## 八、GoogleNet结构讲解，准备用GoogleNet实现图像识别

### 1. 下载模型

​	① 到caffe的github上去下载训练好的GoogleNet模型

​		https://github.com/BVLC/caffe

​	② pad就是给图像补零，pad:2就是补两圈零

​	③ LRN就是局部相应归一化，可以提高模型识别的准确率

​	④ Inception结构中，不同大小的卷积核意味着不同大小的感受野，最后的合并意味着不同尺度特征的融合。采用1，3，5为卷积核的大小，是因为使用步长为1，pad=0,1,2的方式采样之后得到的特征平面大小相同。

​	⑤ concat层用来合并数据

### 2. 准备要识别的图片

​		新建文件夹，放入相应的图片

### 3. 准备synset_words.txt文件

​		内容：编号 物体分类（共一千种）

​		网上下载

### 4. GoogleNet图像识别

```python
# 定义Caffe根目录
caffe_root = ‘E:/caffe-windows/’
# 网络结构描述文件
deploy_file = caffe_root+‘models/bvlc_googlenet/deploy.prototxt’
# 训练好的模型
model_file = caffe_root+‘models/bvlc_googlenet/bvlc_googlenet.caffemodel’

# cpu模式
caffe.set_mode_cpu()

# 定义网络模型
net = caffe.Classifier(deploy_file, # 调用deploy文件
model_file, # 调用模型文件
mean=np.load(caffe_root +‘python/caffe/imagenet/ilsvrc_2012_mean.npy’).mean(1).mean(1), #调用均值文件
channel_swap=(2,1,0), # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
raw_scale=255, #python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
image_dims=(224, 224)) # 输入模型的图片要是224*224的图片

# 分类标签文件
imagenet_labels_filename = caffe_root +‘models/bvlc_googlenet/synset_words.txt’
# 载入分类标签文件
labels = np.loadtxt(imagenet_labels_filename, str, delimiter=’\t’)

# 对目标路径中的图像，遍历并分类
for root,dirs,files in os.walk(caffe_root+‘models/bvlc_googlenet/image/’):
	for file in files:
		# 加载要分类的图片
		image_file = os.path.join(root,file)
		input_image = caffe.io.load_image(image_file)
        
        # 打印图片路径及名称
        image_path = os.path.join(root,file)
        print(image_path)
        
        # 显示图片
        img=Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        # 预测图片类别
        prediction = net.predict([input_image])
        print 'predicted class:',prediction[0].argmax()

        # 输出概率最大的前5个预测结果
        top_k = prediction[0].argsort()[-5:][::-1]
        for node_id in top_k:     
            #获取分类名称
            human_string = labels[node_id]
            #获取该分类的置信度
            score = prediction[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
```



## 九、Caffe在windows下GPU版本的安装

### 1. 安装CUDA

​	准备好NVIDIA的显卡，下载并安装CUDA，

​	https://developer.nvidia.com/cuda-downloads

​	安装好之后把CUDA安装目录下的bin和lib\x64添加到path环境变量中。

### 2. Cudnn下载安装

​	https://developer.nvidia.com/rdp/cudnn-download

​	解压压缩包，把压缩包中bin，include，lib中的文件分别拷贝到C:\Program Files\Nvidia GPU Computing\Toolkit\CUDA\v8.0目录下对应的目录中。

### 3. 修改CommonSetting.props文件

​		<CpuOnlyBuild>false</CpuOnlyBuild>

​		<UseCuDNN>true</UseCuDNN>

​		<CudaVersion>8.0</CudaVersion>

​		<PythonSupport>false</PythonSupport>

### 4. 打开Caffe.sln

​	先编译libcaffe

​	再编译整个工程。

### 5. 测试

​	用GPU的方式运行MNIST数据集分类。

## 十、自定义网络训练数据

### 1. 准备数据集

​	给大家提供一下图片数据下载地址

​	animal:http://www.robots.ox.ac.uk/~vgg/data/pets

​	flower:http://www.robots.ox.ac.uk/~vgg/data/flowers

​	plane:http://www.robots.ox.ac.uk/~vgg/data/airplanes_side/airplanes_side.tar

​	house:http://www.robots.ox.ac.uk/~vgg/data/houses/houses.tar

​	guitar:http://www.robots.ox.ac.uk/~vgg/data/guitars.tar

### 2. 制作标签

​	格式： 文件路径+标签

```
import os
#定义caffe根目录
caffe_root='E:/graduate_student/deep_learning/caffe/new_Win_caffe/document/1/caffe-windows/caffe-windows'
 
#制作测试标签数据
i=0 #标签
with open(caffe_root + 'models/my_models_recognition/labels/test.txt','w') as test_txt:
    for root,dirs,files in os.walk(caffe_root+'models/my_models_recognition/data/test/'): #遍历文件夹
        for dir in dirs:
            for root,dirs,files in os.walk(caffe_root+'models/my_models_recognition/data/test/'+str(dir)): #遍历每一个文件夹中的文件
                for file in files:
                    image_file = str(dir) + '\\' + str(file)
                    label = image_file + ' ' + str(i) + '\n'       #文件路径+空格+标签编号+换行 
                    test_txt.writelines(label)                   #写入标签文件中
                i+=1#编号加1
                
print "成功生成文件列表"
```

### 3. 数据转换

​	将图片转化为LMDB格式

```
%格式转换的可执行文件%
%重新设定图片的大小%
%打乱图片%
%转换格式%
%图片路径%
%图片标签%
%lmdb文件的输出路径%
 
E:\graduate_student\deep_learning\caffe\new_Win_caffe\document\1\caffe-windows\caffe-windows\Build\x64\Release\convert_imageset.exe ^
--resize_height=256 --resize_width=256 ^
--shuffle ^
--backend="lmdb" ^
E:\graduate_student\deep_learning\caffe\new_Win_caffe\document\1\caffe-windows\caffe-windows\models\my_models_recognition\data\test\ ^
E:\graduate_student\deep_learning\caffe\new_Win_caffe\document\1\caffe-windows\caffe-windows\models\my_models_recognition\labels\test.txt ^
E:\graduate_student\deep_learning\caffe\new_Win_caffe\document\1\caffe-windows\caffe-windows\models\my_models_recognition\lmdb\test\ %ttest文件不需要提取创建%
pause
```

### 4. 修改网络模型文件

```
name: "CaffeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
#  transform_param { #求均值方法1
#    mirror: true
#    crop_size: 227
#    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
#  }
 
#mean pixel / channel-wise mean instead of mean image
  transform_param { #求均值方法2
    crop_size: 227  #随机截取227*227，框选小的图片，增大数据集
    mean_value: 104 #大量图片求的均值
    mean_value: 117
    mean_value: 123
    mirror: true	#镜像
  }
  data_param {
    source: "D:/Development/caffe/caffe-master/caffe-master/models/my_models_recognition/lmdb/train"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
#  transform_param {
#    mirror: false
#    crop_size: 227
#    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
#  }
 
# mean pixel / channel-wise mean instead of mean image
  transform_param {
    crop_size: 227
    mean_value: 104
    mean_value: 117
    mean_value: 123
    mirror: false
  }
  data_param {
    source: "D:/Development/caffe/caffe-master/caffe-master/models/my_models_recognition/lmdb/test"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 512
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 5
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc8"
  bottom: "label"
  top: "loss"
}
```

### 5. 修改超参数文件

```
net: "D:/Development/caffe/caffe-master/caffe-master/models/my_models_recognition/train_val.prototxt"
test_iter: 30 #1000 /batch
test_interval: 200  #间隔数
 
base_lr: 0.01
lr_policy: "step"  #每隔stepsize步，对学习率进行改变
gamma: 0.1
stepsize: 1000
 
display: 100    #每100次显示一次
max_iter: 5000
momentum: 0.9
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "D:/Development/caffe/caffe-master/caffe-master/models/my_models_recognition/model"
solver_mode: CPU
```

### 6. 训练模型

```
 %train训练数据%
 %超参数文件%
 
D:\Development\caffe\caffe-master\caffe-master\Build\x64\Release\caffe.exe train ^
-solver=D:/Development/caffe/caffe-master/caffe-master/models/my_models_recognition/solver.prototxt
 
pause
```

==loss没有改变：修改优化器，使用不同的优化器查看结果==

### 7. 测试模型

​	① 修改网络结构

```
name: "CaffeNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }	 #dim:1 修改
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  inner_product_param {
    num_output: 512  #修改，原因自己电脑的显存或cpu有限
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  inner_product_param {
    num_output: 512	  #修改，原因自己电脑的显存或cpu有限
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8"
  inner_product_param {
    num_output: 5   #修改类别
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8"
  top: "prob"
}
```

​	② 编写python文件测试

```
import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import sys
 
 
#定义Caffe根目录  这个需要根据自己的目录进行修改
caffe_root = 'D:/Development/caffe/caffe-master/caffe-master/'
#网络结构描述文件
deploy_file = caffe_root+'models/my_models_recognition/deploy.prototxt'
#训练好的模型
model_file = caffe_root+'models/my_models_recognition/model/model_iter_5000.caffemodel'
 
#gpu模式
#caffe.set_device(0)
caffe.set_mode_cpu()
 
#定义网络模型
net = caffe.Classifier(deploy_file, #调用deploy文件
                       model_file,  #调用模型文件
                       channel_swap=(2,1,0),  #caffe中图片是BGR格式，而原始格式是RGB，所以要转化
                       raw_scale=255,         #python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
                       image_dims=(227, 227)) #输入模型的图片要是227*227的图片
 
 
#分类标签文件
imagenet_labels_filename = caffe_root +'models/my_models_recognition/labels/label.txt'
#载入分类标签文件
labels = np.loadtxt(imagenet_labels_filename, str)
 
#对目标路径中的图像，遍历并分类
for root,dirs,files in os.walk(caffe_root+'models/my_models_recognition/image/'): #循环每一个文件
    for file in files:
        #加载要分类的图片
        image_file = os.path.join(root,file)
        input_image = caffe.io.load_image(image_file)
 
        #打印图片路径及名称
        image_path = os.path.join(root,file)
        print(image_path)
        
        #显示图片
        img=Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        #预测图片类别
        prediction = net.predict([input_image])
        print 'predicted class:',prediction[0].argmax()
 
        # 输出概率最大的前5个预测结果
        top_k = prediction[0].argsort()[::-1]
        for node_id in top_k:     
            #获取分类名称
            human_string = labels[node_id]
            #获取该分类的置信度
            score = prediction[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
```

## 十一、迁移学习

### 1. 训练好的模型

​	CaffeNet的模型文件下载，在readme.md中选择网址下载

### 2. 修改网络结构文件

```
name: "FlickrStyleCaffeNet"
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  image_data_param {
    source: "examples/money_test/data/train.txt"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"
  }
  image_data_param {
    source: "examples/money_test/data/test.txt"
    batch_size: 50
    new_height: 256
    new_width: 256
  }
}
..........
layer {
  name: "fc8_flickr"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8_flickr"
  # lr_mult is set to higher than for other layers, because this layer is starting from random while the others are already trained
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 17    #这里我们的分类数目
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
.....
```

finetune：微调，主要训练最后一层

### 3. 修改超参数文件

```
net: "examples/money_test/fine_tune/train_val.prototxt"
test_iter: 20
test_interval: 50
base_lr: 0.001
lr_policy: "step"
gamma: 0.1
stepsize: 2000
display: 1
max_iter: 10000
momentum: 0.9
weight_decay: 0.0005
snapshot: 1000
snapshot_prefix: "examples/money_test/fine_tune/finetune_money"
solver_mode: CPU
```

### 4. 批处理文件

```
./build/tools/caffe train -solver examples/money_test/fine_tune/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

## 十二、Snapshot使用

​	需要继续训练网络

​	solverstate：参数状态文件

​	在执行语句中加入参数：-snapshot=xxx.solverstate

