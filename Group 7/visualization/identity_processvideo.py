import cv2
import mediapipe as mp
import time
from tqdm import tqdm

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

def process_frame(img):
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for i in range (33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            if i == 0: #鼻尖
                img =  cv2.circle(img, (cx, cy), 10, (0,0,255), -1)
            elif i in [11, 12]: #肩膀
                img = cv2.circle(img, (cx, cy), 10, (223, 155, 6), -1)
            elif i in [23, 24]: #髋关节
                img = cv2.circle(img, (cx, cy), 10, (1, 240, 255), -1)
            elif i in [13, 14]:
                img = cv2.circle(img, (cx, cy), 10, (140, 47, 240), -1)
            elif i in [25, 26]:
                img = cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            elif i in [15, 16, 27, 28]:
                img = cv2.circle(img, (cx, cy), 10, (223, 155, 60), -1)
            elif i in [17, 19, 21]:
                img = cv2.circle(img, (cx, cy), 10, (94, 128, 121), -1)
            elif i in [18, 20, 22]:
                img = cv2.circle(img, (cx, cy), 10, (16, 144, 247), -1)
            elif i in [27, 29, 31]:
                img = cv2.circle(img, (cx, cy), 10, (29, 123, 243), -1)
            elif i in [28, 30, 32]:
                img = cv2.circle(img, (cx, cy), 10, (193, 182, 255), -1)
            elif i in [9, 10]:
                img = cv2.circle(img, (cx, cy), 10, (205, 235, 255), -1)
            elif i in [1, 2, 3, 4, 5, 6, 7, 8]:
                img = cv2.circle(img, (cx, cy), 10, (94, 218, 121), -1)
            else:
                img = cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)
    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100*scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scaler, (255,0,0),3)
    end_time = time.time()
    fps = 1/(end_time - start_time)
    scaler = 1
    img = cv2.putText(img, 'FPS' + str(int(fps)), (25 * scaler, 50*scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25*scaler, (255,0,0),3)
    return img

def generate_video(input_path='F:/human identity/video/4.mp4'):
    filedhead = input_path.split('/')[-1]
    output_path = "out-" + filedhead
    print('视频开始处理', input_path)
    cap = cv2.VideoCapture(input_path)
    frame_count = 0

    # VideoCaputre对象是否成功打开
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 视频解码，解码器选成mp4格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    with tqdm(total=frame_count - 1) as pbar:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                try:
                    frame = process_frame(frame)
                except:
                    print('Error')
                    pass
                if success is True:
                    out.write(frame)
                    pbar.update(1)
        except:
            print('中途中断')
            pass
    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)

if __name__ == "__main__":
    generate_video(input_path='F:/human identity/video/4.mp4')