# 准备环境
* Python 3.5
* Pytorch 1.3
* OpenCV 4.0
* Numpy 1.15
* TensorboardX
* Apex
# 准备数据集
* 将下载的数据集保存到data文件下
# 下载模型
* 想测试F3Net的性能，请下载模型到out文件下
* 想训练你自己的模型，请下载预训练模型到res文件下
# 训练
```
cd src/
python3 train.py
```
* 训练之后，得到的训练模型会保存到out文件下
# 测试 
```
cd src
python3 test.py
```
* 测试之后，运行后的显著图会被保存到eval/F3Net/文件下
# 评估
* 想评估F3Net的性能，请使用MATLAB运行main.m
```
cd eval
matlab
main
```
# 引用
```
@inproceedings{F3Net,
  title     = {F3Net: Fusion, Feedback and Focus for Salient Object Detection},
  author    = {Jun Wei, Shuhui Wang, Qingming Huang},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2020}
}
```
上面提到的数据集以及模型来自这篇论文，大家如有需要可以看这篇论文以及相应github上获取 
