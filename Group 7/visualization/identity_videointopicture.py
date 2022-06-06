import cv2
import os
import glob
import mediapipe as mp
from tqdm import tqdm

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
def video_into_picture(video_path = 'F:/human identity/main/out-6.mp4',
                       output_path = 'F:/human identity/pic/',
                       interval = 1):
    file_name = glob.glob("F:/human identity/pic/*")
    # 检测文件夹里面是否为空，如为空，则file_name = []， 进入if判断
    if not file_name:
        print("File doesn't exit")
    else:
        for file in file_name:
            os.remove(file)
        print("所有文件删除成功")
    num = 1
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)#帧率
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)#总的帧数
# 打印一下（以上三行可以不要，写在这里是为了大概知道视频会输出多少张图片来合理的取帧的间隔）
    Frames = int(frames)
    print("视频的帧率是", int(fps),"总帧数为", Frames)
    lmlist = []
    with tqdm(total=frames - 1) as pbar:
        try:
            while video.isOpened():
                is_read, frame = video.read()
                if is_read:
                    try:
                        if num % interval == 1 or num % interval == 0:
                            file_name = '%d' % num #给转换的图片进行命名
                            cv2.imwrite(output_path + str(file_name) + '.jpg', frame)  # 以.jpg的格式输出图片，保存在指定的目录下

                            img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(img_RGB)
                            for id, lm in enumerate(results.pose_landmarks.landmark):
                                lmlist.append((int(file_name), id, lm.x, lm.y, lm.z))
                            cv2.waitKey(1)
                        num = num + 1
                    except:
                        print('Error')
                else:
                    break
                if is_read is True:
                    pbar.update(1)
        except:
            print('中途中断')
            pass
    print("按照"+ str(interval) + "的帧数将" + "视频转换为图片输出完成")
    return lmlist
# 测试函数
if __name__ == '__main__':
    video_into_picture()