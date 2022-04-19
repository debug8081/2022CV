import cv2
import mediapipe as mp
import time
#初始化数据，设定数据初值
import mediapipe.python.solutions.pose

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('F:/human identity/video/1.mp4')
pTime = 0

#进入循环，读取视频
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #将姿态检测的点打印出来，显示在输出栏上
    print(results.pose_landmarks)
    #将姿态检测的点连成线
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    #计算视频的fps值，并反馈到视频上
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    #设置窗口名称为Image，调整大小为640*480
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 640, 480)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # 点击窗口x即可关闭窗口，结束播放视频
    if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        break
cap.release()
cv2.destroyAllWindows()
