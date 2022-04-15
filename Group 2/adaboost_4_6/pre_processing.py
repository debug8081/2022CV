"""
图片预处理,
分别提取两类图片的LBP特征并添加标签
将特征及标签存入Excel表格中
"""
import os
import numpy as np
from PIL import Image
from model import LocalBinaryPatterns
import pandas as pd
import openpyxl


# 实例化一个LBP特征模型
desc = LocalBinaryPatterns(8, 2)


def getdata(path, label):
    """
    图像处理函数
    :param path: 训练文件地址
    :param label: 该组图像的标签
    :return: 图片LBP特征及对应的标签
    """
    # 创建人脸数据存储列表和标签列表
    datas = list()
    # 提取图片路径信息
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagepath in imagepaths:
        img = Image.open(imagepath).convert("L")  # 以灰度方式打开图片
        img = img.resize((24, 24))
        lbp = desc.describe(img).ravel()  # 获取图像的LBP特征并展平为一维数据
        # lbp = pd.DataFrame(lbp)  # 将ndarray格式转换为DataFrame
        # 使用字典存储lbp特征及标签
        data = {
            "label": label,
            "lbpdata": lbp,
        }
        datas.append(data)
    return datas


if __name__ == "__main__":
    datas = pd.DataFrame()
    ids = list()
    # 数据路径
    path1 = "./data/faces"
    path2 = "./data/things"
    # lbp特征及对应标签
    lbpdatas1 = getdata(path1, 1)  # 人脸lbp数据, 添加标签为1
    lbpdatas2 = getdata(path2, -1)  # 非人脸lbp数据, 添加标签为-1
    lbpdatas = lbpdatas1 + lbpdatas2  # 组合两个LBP特征记录表
    keys = list(lbpdatas[0].keys())
    for row in range(0, len(lbpdatas)):
        for col, key in zip(range(len(keys)), keys):
            if col % 2 == 0:
                ids.append(lbpdatas[row][key])
            else:
                a = lbpdatas[row][key].reshape(1, len(lbpdatas[row][key]))
                data = pd.DataFrame(a)
                datas = pd.concat([datas, data])

    np.savetxt("lbpdata.txt", datas, fmt="%.1f")
    np.savetxt("lbpID.txt", ids, fmt="%d")
    # with open("lbpID.txt", "w") as fp:
    #     [fp.write(str(item) + '\n') for item in ids]


