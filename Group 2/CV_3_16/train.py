import os
import cv2 as cv
from PIL import Image
import numpy as np


def getdata(path):
    """
    训练函数
    :param path: 训练文件地址
    :return: 人脸数据及对应的人脸ID
    """
    # 创建人脸数据存储列表和对应的ID存储列表
    facedata = list()
    ids = list()
    # 提取图片信息
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 加载分类器
    face_detector = cv.CascadeClassifier(
        "./haarcascade/haarcascade_frontalface_alt2.xml")
    # 训练
    for imagepath in imagepaths:
        PIL_img = Image.open(imagepath).convert("L")  # 以灰度方式打开图片
        img_numpy = np.array(PIL_img, "uint8")  # 向量化
        face = face_detector.detectMultiScale(img_numpy, 1.1, 5)  # 获取人脸特征
        ID = int(os.path.split(imagepath)[1].split(".")[0])
        for x, y, w, h in face:
            ids.append(ID)
            facedata.append(img_numpy[y:y+h, x:x+w])
    # print(ID)
    # print(facedata)
    return facedata, ids


if __name__ == "__main__":
    # 训练数据路径
    path = "./data/chunwan/"
    # 人脸及对应名称
    facedata, ids = getdata(path)
    # 识别器加载
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(facedata, np.array(ids))
    # 保存训练文件
    recognizer.write("./trainer/train_data.yml")
    print("保存完成")
