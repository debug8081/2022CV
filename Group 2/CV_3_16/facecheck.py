import cv2 as cv


def face_check(img):
    """
    人脸检测函数
    :param img:要检测的图片
    :return:人脸位置信息,
    """
    # 灰度化
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_classifiter = cv.CascadeClassifier(r"D:\OpenCV\opencv\sources\data"
                                            r"\haarcascades\haarcascade_frontalface_default.xml")
    # 检测
    face = face_classifiter.detectMultiScale(gray_img)
    return face


img = cv.imread("./face/face3.jpg")
face_data = face_check(img)
# 使用红色框标记
for x, y, w, h in face_data:
    cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)

cv.imshow("result", img)

while True:
    # 按下q退出
    if ord("q") == cv.waitKey(0):
        break

# 释放缓存
cv.destroyAllWindows()
