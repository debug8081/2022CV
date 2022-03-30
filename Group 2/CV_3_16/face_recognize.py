import cv2 as cv
import os

# file_list = list()  # 存储处理好的图片, 生成演示视频用

# 加载识别器
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("./trainer/train_data.yml")

# 加载任务姓名
names = list()
path = './data/chunwan/'
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
for imagePath in imagePaths:
    name = str(os.path.split(imagePath)[1].split('.', 2)[1])
    names.append(name)


def face_recognize(img):
    """
    进行人脸识别
    :param img: 识别图片
    :return: 标记好的图片
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_classifiter = cv.CascadeClassifier(
        "./haarcascade/haarcascade_frontalface_alt2.xml")
    # 检测
    face = face_classifiter.detectMultiScale(gray_img, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (200, 200), (400, 400))
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)  # 框选人脸
        ID, confidence = recognizer.predict(gray_img[y:y + h, x:x + w])
        if confidence > 50:
            cv.putText(img, "unknown", (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv.putText(img, str(names[ID - 1]), (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv.imshow("result", img)
    return img


cap = cv.VideoCapture('test.mp4')
while True:
    flag, frame = cap.read()  # 按帧读取图片
    # 视频播放完成退出循环
    if not flag:
        break
    face_recognize(frame)  # 识别人脸

    # 按空格键直接结束识别
    if ord(' ') == cv.waitKey(10):
        break

"""
    # 生成演示视频文件
    img_res = face_recognize(frame)
    img_res = cv.resize(img_res, (1280, 720))
    file_list.append(img_res)
fps = 24
video = cv.VideoWriter("VideoTest1.mp4", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, (1280, 720))
for item in file_list:
    video.write(item)
video.release()
"""

# 释放内存
cv.destroyAllWindows()
cap.release()