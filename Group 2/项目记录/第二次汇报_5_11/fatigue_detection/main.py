import cv2

from model import mtcnn


# file_list = list()  # 存储处理好的图片, 生成演示视频用

model = mtcnn()  # 创建网络
threshold = [0.5, 0.6, 0.7]  # 检测阈值


img = cv2.imread('test.jpg')
temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rectangles = model.detectFace(temp_img, threshold)  # 检测图片

draw = img.copy()
for rectangle in rectangles:
    W = int(rectangle[2]) - int(rectangle[0])
    H = int(rectangle[3]) - int(rectangle[1])

    # 标记人脸
    cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 2)

    # 标记特征点
    for i in range(5, 15, 2):
        cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)

cv2.imwrite("out.jpg", draw)
cv2.imshow("test", draw)
c = cv2.waitKey(0)

"""
# 合成视频
cap = cv2.VideoCapture('test.mp4')
while True:
    flag, frame = cap.read()  # 按帧读取图片
    # 视频播放完成退出循环
    if not flag:
        break
    img = frame
    temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rectangles = model.detectFace(temp_img, threshold)  # 检测图片

    for rectangle in rectangles:
        W = int(rectangle[2]) - int(rectangle[0])
        H = int(rectangle[3]) - int(rectangle[1])

        # 标记人脸
        cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (0, 0, 255), 2)

        # 标记特征点
        for i in range(5, 15, 2):
            cv2.circle(img, (int(rectangle[i + 0]), int(rectangle[i + 1])), 1, (255, 0, 0), 4)

    img_res = img
    cv2.imshow("test", img)

    # 按空格键直接结束识别
    if ord(' ') == cv2.waitKey(10):
        break

    # 生成演示视频文件
    img_res = cv2.resize(img_res, (1280, 720))
    file_list.append(img_res)
fps = 24
video = cv2.VideoWriter("VideoTest1.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (1280, 720))
for item in file_list:
    video.write(item)
video.release()

cv2.destroyAllWindows()
cap.release()
"""
