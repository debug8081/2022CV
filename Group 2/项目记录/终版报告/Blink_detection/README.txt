# 整体说明
眨眼检测项目代码。主要用到MTCNN网络和CNN网络。前者作用为提取人脸位置和人脸五点检测数据，
CNN网络用于判断眨眼情况。

# 文件标注
data文件夹存放训练数据，数据集为CEW闭眼数据集。通过数据增强扩充至14538张图片，
并按照0.3的比例抽取部分图片作为测试集，要使用需要解压。原始数据集压缩包在文件夹data1中

image文件夹存放了CNN网络的损失函数图像和CNN网络结构

log文件夹存放训练好的CNN网络模型，文件过大，这里使用网盘链接传输
（链接：https://pan.baidu.com/s/1E1qkLjpHjW0gelh0OnN6AQ?pwd=1111 
，链接可自动填充提取码，未成功的话提取码：1111，下载后放在项目文件夹下即可）

model_data文件夹存放的MTCNN网络的模型。

cnn_check.py文件为CNN网络测试文件，该文件运行了测试集中所有数据，统计预测正确率，
并且返回预测错误图像。

CNNtest.py是单一图片的CNN预测文件。（可以直接运行，log文件夹中已经包含训练好的模型）
CNNtrain.py是CNN网络的训练文件。（训练用数据集需要解压）

model.py是MTCNN网络的模型文件

modelcnn.py是CNN网络的模型文件。

other_func.py中定义了项目中用到的一些函数。

video.mp4和result.mp4为效果展示文件。

# requirements
requirements.txt文件为项目中用到的包。
