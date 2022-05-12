# 整体内容
使用MTCNN网络进行人脸位置及关键点检测。代码为MTCNN网络模型结构。

# 文件夹内容
model_data文件夹存放了训练好的模型。main.py文件是项目入口，使用时直接运行即可。model.py文件为网络结构文件。other_func.py是
项目中用到的一些函数。两张图片文件为测试文件及结果。requirements.txt文件为所需库及版本。


# TensorFlow1.13.1安装
该项目运行所需组合为CUDA10.0 + cuDNN7.6.1 + python3.7 + tensorflow-gpu1.13.1

请注意自己下载的版本要对应，对应，对应。不能高也不能低。如果想要使用其他版本TensorFlow，请查看对应版本的CUDA和cuDNN

TensorFlow安装教程连接如下：
https://blog.csdn.net/qq_43351106/article/details/96467497

教程中安装CUDA部分的链接失效，可以使用如下链接进行安装：
https://blog.csdn.net/qq_40923413/article/details/108070052

# 特别注意
TensorFlow安装教程中的安装TensorFlow部分，不建议直接将清华源至anaconda的下载源中，本人添加后下载其他库时会有请检查hostname的提示（不知道为啥，确认没开代理）。
直接使用pip的-i指定下载源即可。

CUDA安装过程中，可执行文件的提示和CUDA教程不太一样，不一样的直接略过即可，将有的选项调整好就可以。

至于anaconda安装部分网上很多经验帖都可以用，但是不建议从官网下载，很慢很慢，建议从镜像下载安装包。

# 其他库安装
不要直接install requirements.txt。先安装好TensorFlow1.13.1后，安装requirements.txt中除了numpy外的剩余库，numpy库会随着TensorFlow安装而安装。
但是安装的numpy库为最新版本，会导致TensorFlow运行时报警告，但是修改为适配版本后其他库不能正常运行，这一点无法避免（或许有，没想到）。警告无法避免，
但不影响运行。

# 使用方式
直接运行main.py文件即可。
