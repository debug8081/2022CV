import cv2
import mediapipe as mp
import time
#初始化数据，设定数据初值
import mediapipe.python.solutions.pose
import self as self

class Dector():
    #创建类别，将其命名为Dector
    def __init__(self, mode= False, complexity= 1, landmarks= True, enableSeg= False,
                 smoothSeg= True, detectionCon= 0.5, trackingCon= 0.5):
        #这些是初始化pose函数里面的指标，通过按ctrl+左键pose = mpPose.Pose()里面的pose，可以得到检测函数的初始值
        self.mode = mode
        self.complexity = complexity
        self.landmarks = landmarks
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
#这是初始化medpipe库中的检测函数
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.landmarks, self.enableSeg,
                                     self.detectionCon, self.trackingCon)
#findpose是检测人体姿态，并且把检测到的每个点连起来
    def findpose(self, img, draw= True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
#findpostion是我们要具体到某个点，把它进行放大处理，或者找到其详细坐标定位
    def findpostion(self, img, draw= True):
        lmlist = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmlist
#定义主函数main
def main():
    cap = cv2.VideoCapture('F:/human identity/video/3.mp4')
    pTime = 0
    de = Dector()
    while True:
        success, img = cap.read()
        # 进入循环，读取视频
        img = de.findpose(img)
        lmlist = de.findpostion(img)
        #打印出某一个点的x，y值坐标，例如14
        print(lmlist[14])
        #这里如果要放大14号点的位置，可以采取一下方法
        #将lmlist = de.findpostion(img)修改为lmlist = de.findpostion(img，draw= False)意为不执行画点操作
        # 然后输入cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0, 0, 0), cv2.FILLED)，
        # 是把cx，cy都替换成了14号点位的【1】【2】号坐标，也就是14号位的x，y轴的实时坐标，
        # 大小更改为了15，颜色改成了黑色（颜色可以自调），在findpose里面默认用的是红色（0,0,255）


        # 计算视频的fps值，并反馈到视频上
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # 设置窗口名称为Image，调整大小为640*480
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1280, 780)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        # 点击窗口x即可关闭窗口，结束播放视频
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()
#定义程序入口
if __name__ == "__main__":
     main()