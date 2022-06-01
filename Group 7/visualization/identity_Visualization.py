# 可视化步骤：将视频处理为带33个关键的点待检测视频->将待检测视频逐帧转化为图片，并存入指定目录
# ->将指定目录下的图片进行目标点检测，生成具有[关键点位，坐标x， 坐标y， 坐标z]的数据
# ->将数据可视乎，并将可视化模块嵌入进视频中->通过可视化模板，进行阈值筛选，从而计数
import cv2
import time
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from identity_videointopicture import video_into_picture
from identity_processvideo import generate_video
from Chinese import cv2AddChineseText
class Visualization:
    #设置初始值，定义两个逻辑值变量均为0，用于计数
    def __init__(self,logical_up=0, logical_down=0, count=0):
        self.logical_up = logical_up
        self.logical_down = logical_down
        self.count = count
    #计算输入视频帧率，返回值是视频的帧率
    def count_frames(self, video_path):
        video = cv2.VideoCapture(video_path)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = int(frames)
        return frames
    #通过视频进行姿态估计， 将33个关键点坐标生成csv文件，无返回值
    def write_csv(self, lmlist):
        text = ['picture_name', 'points_id', 'x', 'y', 'z']
        with open('landmark.csv', 'w', newline="") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(text)
            csvwriter.writerows(lmlist)
        print("csv文件生成完成")
        # 读取csv文件,获取某一个关键点的坐标集合
    #读取csv文件，并且提取某一个关键点的值，返回值是特点关键点的y轴坐标
    def readcsv_get_landmarks(self, csv_path, points):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)
            lm = []
            # enumerate()获取每个元素的索引及其值
            # for index, column_header in enumerate(header_row):
            #     lm.append([index, column_header])
            for row in reader:
                if row[1] == str(points):
                    lm.append(row)
            lm1 = []
        for i in lm:
            res = list(map(float, i))
            lm1.append(res)
        x, y, z = [], [], []
        for i in lm1:
            x.append(i[2])
            y.append(i[3])
            z.append(i[4])
        return y
    #通过读取某一个关键点的坐标集合，生成逐帧坐标图片，无返回值
    def get_plt_picture(self, frames, landmark_lmlist):
        frame = np.arange(1,frames+1)
        fig = plt.figure()
        plt.xlim(xmin=0, xmax=frames)  # x轴的范围[0,frames]
        plt.ylim(ymin=0, ymax=1)  # y轴的范围[0,2]
        plt.xlabel('X')
        plt.ylabel('Y')
        # 调整x轴刻度
        plt.xticks(np.linspace(0, frames, 10))
        # 调整y轴刻度
        plt.yticks(np.linspace(0, 1, 11))
        print("现在生成二维坐标图片")
        with tqdm(total=frames - 1) as pbar:
            try:
                for i in frame:
                    if i == 1:
                        ax = plt.gca()
                        plt.scatter(frame[0], landmark_lmlist[0], color="red")
                        plt.savefig('F:/human identity/plt/' + str(i) + '.jpg')
                        pbar.update(1)
                    else:
                        ax = plt.gca()
                        plt.plot(frame[0:i], landmark_lmlist[0:i], color="red")
                        plt.savefig('F:/human identity/plt/' + str(i) + '.jpg')
                        pbar.update(1)
            except:
                print('中途中断')
                pass
        print("二维坐标图片生成完毕，已保存")
    #将pic文件夹下的图片与plt_pic下的图片进行整合，合并为一个图片，存储在plt+pic文件夹下
    def paste_picture(self, frames, input_path1='F:/human identity/pic/'
                      ,input_path2='F:/human identity/plt/'
                      ,output_path='F:/human identity/plt+pic/'):
        print("准备合并图片")
        with tqdm(total=frames - 1) as pbar:
            for i in range(frames):
                try:
                    img1 = Image.open(input_path1 + str(i+1) + '.jpg')
                    img2 = Image.open(input_path2 + str(i+1) + '.jpg')
                    img1.paste(img2,(0,0))
                    img1.save(output_path + str(i+1) + '.jpg')
                    pbar.update(1)
                except:
                    print('中途中断')
                    pass
        print("图片合成完毕，已保存")
    #将plt_pic文件夹下所有的图片转换为一个视频：无返回值
    # a. 使用循环获取路径下的所有图片；
    # b. cv2.imread()读取所有图片；
    # c. 将读取的图片存于列表中；
    # d. 使用cv2.VideoWriter()创建VideoWriter对象，注意参数的设置；
    # e. 使用cv2.VideoWriter().write()保存每一帧图像到视频文件；
    # f. 释放 VideoWriter对象；
    def picture_into_video(self, name, frames
                           ,input_path='F:/human identity/plt+pic/'
                           ,output_path='F:/human identity/video/'):
        img_array = []
        file = []
        for num in range(frames):
            fname = input_path + str(num+1) + '.jpg'
            file.append(fname)
            # file = sorted(glob.glob(input_path + '*.jpg'), key=os.path.getsize)
        print("准备将图片转换为视频，现生成图像数组")
        with tqdm(total=frames - 1) as pbar:
            for filename in file:
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width, height)
                img_array.append(img)
                pbar.update(1)
        print("图像数组生成完毕，下面转换为视频")
        with tqdm(total=frames - 1) as pbar:
            out = cv2.VideoWriter(output_path + str(name) +'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            for i in range(len(img_array)):
                out.write(img_array[i])
                pbar.update(1)
            out.release()
        print("视频已生成")
    #读取视频，进行计数
    def put_number_into_video(self, video_path, frames, lmlist_shou, lmlist_jian, lmlist_zhou, lmlist):
        pTime = 0
        cap = cv2.VideoCapture(video_path)
        for i in range(frames):
            success, img = cap.read()
            if success == False:
                break
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            if lmlist[i]<lmlist_shou[i] and lmlist_jian[i]<lmlist_zhou[i]:
                self.logical_up = 1
            elif lmlist[i]>lmlist_zhou[i] and self.logical_up:
                self.logical_down = 1

            if self.logical_up and self.logical_down:
                self.count += 1
                self.logical_up, self.logical_down = 0, 0

            # 将计数结果实时反馈至屏幕上
            cv2.putText(img, "FPS:", (1600, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(img, str(int(fps)), (1700, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            # cv2.putText(img, "The number is :", (0, 100), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 0), 3)
            cv2.putText(img, str(int(self.count)), (1700, 250), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            #由于cv2.putText在输入中文后显示了？？？，不能输出中文，故在Chinese.py文件中封装函数，在对其使用，起到输入中文的目的
            img = cv2AddChineseText(img, "引体向上数量为：", (1200, 300), (255, 0, 0), 60)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

            # 点击窗口x即可关闭窗口，结束播放视频
            if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    #可视化步骤（针对视频）：
    # 1、先将原视频文件进行函数generate_video（），将原视频变成含有33个关键点的视频
    # 2、再将含有33个关键点的视频进行函数video_into_picture()，将视频逐帧处理为图片，并存入pic文件夹中
    # 3、由于函数video_into_picture()这个函数会返回一个带有33个关键点坐标的lmlist列表变量，我们接下来使用write_csv将lmlist列表写入为一个csv文件
    # 4、将生成的csv文件通过函数readcsv_get_landmarks()，获取某个具体的关键点的坐标集合（在视频流下），这会返回一个关键点纵坐标的列表
    # 5、通过使用get_plt_picture()函数读取某一个关键点的坐标集合，生成逐帧坐标图片，无返回值，并储存在plt_pic中
    # 6、再使用paste_picture（）函数将坐标集合进行绘图，横坐标是帧数，纵坐标是关键点的纵坐标，将所有图片储存，储存在plt+pic中
    # 7、再使用picture_into_video（）函数将plt+pic文件下的图片进行转换，储存为视频
    # 8、将最终得到的视频进行put_number_into_video（）进行计数，并实时显示在视频上
    video_path = 'F:/human identity/video/4.mp4'
    pic_path = 'F:/human identity/pic/'
    path = 'F:/human identity/video/'
    generate_video(video_path)#无返回值，生成out-4.mp4视频文件
    lmlist = video_into_picture(video_path, pic_path, 1)#按照1帧将out-4.mp4转换成图片，存储在pic_path中，有返回值，返回一个列表
    vis = Visualization()
    vis.write_csv(lmlist)#将返回的lmlist列表生成一个csv文件，取名为landmark.csv
    frame = vis.count_frames(video_path)#统计视频总帧数，返回值为总帧数
    lmlist_zui = vis.readcsv_get_landmarks('landmark.csv', 9)#获取嘴关键点的纵坐标，返回值为坐标集合列表
    lmlist_jian = vis.readcsv_get_landmarks('landmark.csv', 11)#获取肩关键点的纵坐标，返回值为坐标集合列表
    lmlist_zhou = vis.readcsv_get_landmarks('landmark.csv', 13)#获取肘关键点的纵坐标，返回值为坐标集合列表
    lmlist_shou = vis.readcsv_get_landmarks('landmark.csv', 19)#获取手关键点的纵坐标，返回值为坐标集合列表
    vis.get_plt_picture(frame, lmlist_zui)#无返回值，将嘴部关键点纵坐标进行绘图，横坐标为帧数，纵坐标为嘴部的坐标，并将图片储存在plt_pic中
    vis.paste_picture(frame)#无返回值，利用图片合并，将pic文件夹下的图片与plt_pic下的图片进行整合，合并为一个图片，存储在plt+pic文件夹下
    vis.picture_into_video('output', frame)#无返回值，将plt_pic文件夹下所有的图片转换为一个视频，输出为output.mp4
    vis.put_number_into_video(path + 'output.mp4', frame, lmlist_shou, lmlist_jian, lmlist_zhou, lmlist_zui)#无返回值，将output.mp4视频进行实时计数
    print("已完成")
