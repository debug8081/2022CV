# 项目说明
项目中包含三个代码文件，其中facecheck.py是独立的人脸检测程序，可用于检测图片中人脸位置。

train.py与face_recognize.py文件是人脸识别程序，train.py用于识别器的训练，
face_recognize.py用于识别，test视频是用于识别的素材。

data文件夹中包含chunwan文件夹，用于存储要识别的人脸照片

face文件夹中包含三张人脸照片，用于facecheck.py文件的运行

haarcascade文件夹存放程序中用到的OpenCV提供的两个分类器

VideoTest1是test视频经过识别后生成的结果视频，由于文件过大没有上传成功，
将face_recognaze.py中的关于生成视频文件的注释释放即可在本地生成演示视频

face_recognize.pptx是视频识别的演示文件

# requirements
you can install all the requirements by executing below.

```
# in <path-to-this-repo>
pip install -r requirements.txt
