# Caffe环境配置及安装

# LINUX版

## 一、Caffe环境配置

### 1. 安装环境

#### 	1.1 硬件

​			运行机器：服务器or个人电脑

​			GPU类型：Tesla_V100-SXM2-32GB 

#### 	1.2 系统

​			linux系统：Ubuntu常用的版本有：**16.04（本文档采用）**、14.04。

#### 	1.3 GPU驱动

​			由于使用Nvidia显卡，因此安装CUDA9.2进行驱动控制，关于CUDA的安装可以自行解决。

​			[Nvidia驱动CUDA下载地址](https://developer.nvidia.com/cuda-downloads)

### 2. 第三方库

#### 	2.1 依赖包

​			Linux环境后续依赖包安装命令：

```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install git cmake build-essential
```

​			验证上述依赖包安装成功，可以再次运行安装命令。

​			如提示：<u>升级了 0 个软件包，新安装了 0 个软件包，要卸载 0 个软件包，有 94 个软件包未被升级。</u>

​			则安装成功。

#### 	2.2 Cudnn

​			如需在深度学习中使用GPU进行加速计算，则需安装CUDNN库。

​			[Cudnn下载地址](https://developer.nvidia.com/rdp/cudnn-download)

​			安装成功后，可以通过nvcc -V命令验证是否安装成功，安装成功则会显示Cudnn相关安装信息。

#### 	2.3 Opencv

​			安装版本：opencv3.1，地址：[Opencv下载地址](http://opencv.org/releases.html)，下载Sources版本的opencv包。

​			解压源码包到需要安装的位置，通过命令行进入到已解压的文件夹目录下，执行编译：

```bash
mkdir build # 创建编译的文件目录
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8  #编译
```

​			编译成功后安装opencv：sudo make install #安装

​			安装完成后可以通过查看opencv版本验证是否安装成功：pkg-config –modversion opencv

## 二、Caffe安装

### 	1. 版本库安装

#### 			1.1 安装包下载

​					① 直接执行命令：git clone https://github.com/BVLC/caffe.git

​					② 自行下载：[Windows Caffe下载地址](https://github.com/BVLC/caffe/tree/windows) [Caffe源下载地址](https://github.com/BVLC/caffe)

#### 			1.2 配置文件修改

##### 					1.2.1 复制config文件

​							进入caffe目录，将 Makefile.config.example 文件复制一份并更名为 Makefile.config

​							也可以在 caffe 目录下直接调用以下命令完成复制操作：

​							sudo cp Makefile.config.example Makefile.config

##### 					1.2.2 修改cudnn相关

​							将 #USE_CUDNN := 1 修改成：USE_CUDNN := 1

##### 					1.2.3 修改opencv版本

​							将 #OPENCV_VERSION := 3 修改为： OPENCV_VERSION := 3

##### 					1.2.4 修改Python接口

​							将 #WITH_PYTHON_LAYER := 1 修改为 WITH_PYTHON_LAYER := 1

##### 					1.2.5 修改Python路径

```bash
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
#修改为： 
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

#### 				1.3 Makefile文件修改

```bash
将： NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
替换为：NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
将：LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
改为：LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

#### 				1.4 修改host_config文件

​					修改 /usr/local/cuda/include/host_config.h 文件 :

```bash
将 #error-- unsupported GNU version! gcc versions later than 4.9 are not supported!
改为 //#error-- unsupported GNU version! gcc versions later than 4.9 are not supported!
```

#### 				1.5 安装

​					在caffe目录下执行：make all -j8

#### 				1.6 测试

​					执行测试：sudo make runtest -j8

### 	2. Pycaffe接口安装

​		在上一步成功安装 caffe 之后，就可以通过 caffe 去做训练数据集或者预测各种相关的事了，只不过需要在命令行下通过 caffe 命令进行操作，而这一步 pycaffe 的安装以及 notebook 环境配置只是为了更方便的去使用 caffe ，实际上大多数都是通过 python 来操作 caffe 的，而 notebook 使用浏览器作为界面，可以更方便的编写和执行 python 代码。

#### 		2.1 编译pycaffe

​				在caffe目录下，需要编译pycaffe：sudo make pycaffe -j8

#### 		2.2 验证编译成功

​				进入python环境：import caffe

#### 		2.3 配置notebook环境

​		首先要安装python接口依赖库，在caffe根目录的python文件夹下，有一个requirements.txt的清单文件，上面列出了需要的依赖库，按照这个清单安装就可以了。

​		在安装scipy库的时候，需要fortran编译器（gfortran)，如果没有这个编译器就会报错，因此，我们可以先安装一下。

​		首先进入 caffe/python 目录下，执行安装代码：

```bash
sudo apt-get install gfortran
for req in $(cat requirements.txt); do sudo pip install $req; done
```

​		安装完成之后执行：

```bash
sudo pip install -r requirements.txt
```

​		就会看到，安装成功的，都会显示Requirement already satisfied, 没有安装成功的，会继续安装。

​		然后安装 jupyter ：

```bash
sudo pip install jupyter
```

#### 		2.4 运行notebook

​			安装完成之后执行：

```bash
jupyter notebook
或
ipython notebook
```

​			就会在浏览器中打开notebook, 点击右上角的New-python2, 就可以新建一个网页一样的文件，扩展名为ipynb。在这个网页上，我们就可以像在命令行下面一样运行python代码了。输入代码后，按shift+enter运行，更多的快捷键，可点击上方的help-Keyboard shortcuts查看，或者先按esc退出编辑状态，再按h键查看。

## 参考来源

​		1.[Ubuntu16.04 Caffe 安装步骤记录（超详尽）](https://blog.csdn.net/yhaolpz/article/details/71375762)

# Windows版



