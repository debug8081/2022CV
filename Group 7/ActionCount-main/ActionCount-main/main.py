import sys,csv,cv2
from PyQt5.QtWidgets import (QApplication,QMainWindow,QAction,
                            QFileDialog,QMessageBox)
from UI.Ui_ui1 import Ui_MainWindow
import mediapipe as mp
import numpy as np
import computer

class Main(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initMenuBar()
        self.button()
        self.default()

    def initMenuBar(self):  # 菜单栏
        menuBar = self.menuBar() 
        menuFile = menuBar.addMenu('文件(&F)')
        actionExit = QAction('退出(&Q)', self, shortcut='Ctrl+Q', triggered=QApplication.instance().quit)
        menuFile.addAction(actionExit)

    def default(self):  # 一些默认值
        self.filename = []
        self.keypoint1 = {0:[(23,11,23,25),(24,12,24,26)], 1:[(13,11,13,15), (14,12,14,16)], 2:[(11,23,11,13), (12,24,12,14)] ,3:[(25,23,25,27),(26,24,26,28) ],4:[(23,11,23,25),(24,12,24,26)]} # 0 躯干大腿角[L,R], 1 肘关节角[L,R]，2......以此类推
        self.keypoint2 = {1:[(23,11,23,25),(24,12,24,26)]}
        self.angle = {0:[60,110], 1:[130,150], 2:[40,160] ,3:[70,170] ,4:[110,170]}  # 角度范围 0仰卧起坐躯干-大腿角度, 1俯卧撑肘关节角度，2......以此类推
        self.countBtn = False

    def button(self):  # 按钮
        self.pushButton.clicked.connect(self.video)
        self.pushButton_2.clicked.connect(self.cam)
        self.pushButton_3.clicked.connect(self.analyze)
        self.pushButton_4.clicked.connect(self.btn)
    
    def video(self):  # 选择视频
        self.filename = QFileDialog.getOpenFileNames(self, '选择相同动作视频', '', '')[0] #获取导入视频的路径，并将其存入列表filename中，以便获取多个视频的路径
        self.listWidget.clear()
        self.listWidget.addItems(self.filename)

    def analyze(self):  # 开始计数
        fileName = QFileDialog.getSaveFileName(self, 'Save File','result', "CSV Files (*.csv)")
        # print(fileName) 获取一个元组，一共有两个数据，一个是文件路径，一个是文件类型
        if fileName[0]:  #进行if判断，增加容错
            act = self.comboBox.currentIndex()
            # print(act) act = 字典keypoint的key值，如果我选择了仰卧起坐就是0，选择了俯卧撑就是1，选择了引体向上就是2，以此类推
            if act == 1:
                joint1 = self.keypoint1[act]
                joint2 = self.keypoint2[act]
            else:
                joint1 = self.keypoint1[act]
            angleRange = self.angle[act]
            # print(joint) joint = 字典keypoint下对应key值的value值，也就是说，如果key值是2，那么取得就是[(11,23,11,13), (12,24,12,14)]
            # print(angleRange) angleRange = 字典angle下对应key值的value值，也就是说，如果key值是2，那么取得就是[40,160]
            if self.filename != []:  #对filename进行判断，确认文件是否存在
                with open(fileName[0],'w', newline='') as file0:  #进行csv文件的写入
                    write = csv.writer(file0)
                    write.writerow(['名称','数量'])  #csv文件第一行进行备注
                    self.countBtn = True  #
                    # print(self.filename)
                    # print(self.test)
                    for file in self.filename:
                        name = file.split('/')[-1][:-4]  # 视频文件名
                        # print(self.filename) self.filename = [文件路径]
                        # print(file) file = 文件路径
                        print('当前视频：' + name)
                        count = self.count(file, joint1, angleRange)
                        print(count)
                        write.writerow([name,count])
                    saveTxt = '数据文件已保存至' + str(fileName[0])
                    QMessageBox.information(self,'完成',saveTxt)
                    self.countBtn = False
    
    def cam(self):
        url = 0  # 用于捕获电脑摄像头（Windows系统写0，Mac系统写1）
        act = self.comboBox.currentIndex()
        joint1 = self.keypoint1[act]
        if act == 1:
            joint2 = self.keypoint2[act]
        angleRange = self.angle[act]
        self.count(url, joint1, angleRange)

    def btn(self):
        if self.countBtn:
            self.countBtn = False
            self.pushButton_4.setText('开始计数')
        else:
            self.countBtn = True
            self.pushButton_4.setText('停止计数')
            
    def count(self, video, point, angleRange):  # video视频路径， point(0,1,2,3)前4个关键点， range关节角度范围
        #初始化mediapipe
        mp_drawing = mp.solutions.drawing_utils
        myPose = mp.solutions.pose
        poses = myPose.Pose()

        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.2)  # 视频宽度*0.2  用于显示计数
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.2) # 视频高度*0.2
        size = int(height / 30)  # 字体大小
        count = 0  # 计数
        down = False
        up = False
        while True:
            ret,frame = cap.read()
            act = self.comboBox.currentIndex()
            if ret:
                if self.countBtn == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = poses.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:  # 包含姿势地标的“姿势地标”字段。
                        landmarks = results.pose_landmarks.landmark
                        data = np.array([(lm.x, lm.y, lm.z, lm.visibility ) for lm in landmarks])  #   获取坐标并存入numpy数组
                        try:
                            # 选择身体左或右侧
                            left = point[0]  # (13,11,13,15)点序号 这里的point就是上述的joint元组
                            right = point[1]
                            leftP = 0
                            rightP = 0
                            for i in range(4):  #依次遍历这四个关键点，左边右边各四个点，
                                leftP += data[left[i]][3]    #提取左边4个关键点的可视化数值，合计加入leftP
                                rightP += data[right[i]][3]  #提取右边4个关键点的可视化数值，合计加入rightP
                            if leftP <= rightP:
                                value = right
                            else:
                                value = left
                            # 计算身体角度
                            joint1 = round(computer.cal_angle(data[value[0]], data[value[1]], data[value[2]], data[value[3]]),2)  # v1上端中，v2上远，v3下中，v4下远
                            joint2 = round(computer.cal_angle(data[value[0]], data[value[1]], data[value[2]], data[value[3]]), 2)
                            print(joint1)
                            # 判断并计数
                            if act == 1:
                                if joint1 > angleRange[1] and joint2 > 155:
                                    down = True
                                elif down == True and joint1 < angleRange[0]:
                                    up = True
                                if down == True and up == True:
                                    count += 1
                                    down = False
                                    up = False
                            else:
                                if joint1 > angleRange[1]:
                                    down = True
                                elif down == True and joint1 < angleRange[0]:
                                    up = True
                                if down == True and up == True:
                                    count += 1
                                    down = False
                                    up = False

                            # print(count)
                        except Exception as e:
                            print(e.args)
                        # 画图
                        mp_drawing.draw_landmarks(frame,results.pose_landmarks,myPose.POSE_CONNECTIONS)  
                        cv2.putText(frame, str(count), (width, height), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,255), 3)
                    
                cv2.imshow('Press ESC to Exit', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        return count


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 建立application对象
    win = Main()
    win.show()  # 显示窗体
    sys.exit(app.exec_())  # 运行程序