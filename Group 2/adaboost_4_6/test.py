from model import DecisionTreeClassifierWithWeight
from model import LocalBinaryPatterns
import numpy as np
from PIL import Image


imagepath = "./pic_test/face5.jpg"  # 测试图片路径

# 读取lbp特征信息和标签
x_train = np.loadtxt("lbpdata.txt")
y_train = np.loadtxt("lbpID.txt")

# 创建弱分类器对象
classifiers = DecisionTreeClassifierWithWeight()
classifiers.fit(x_train, y_train)  # 训练分类器

desc = LocalBinaryPatterns(8, 2)  # 创建LBP特征模型对象
img = Image.open(imagepath).convert("L")  # 以灰度方式打开图片
img = img.resize((24, 24))
lbp = desc.describe(img).ravel()  # 获取图像的LBP特征并展平为一维数据
lbp = lbp.reshape(1, len(lbp))

# 识别图像中是否有人脸, 1为含有, -1为不含有
print(classifiers.predict(lbp))
