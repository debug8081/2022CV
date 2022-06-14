1.通过Anaconda安装pytorch(当时废了好大劲才安装下来，问同学也不会)
2.把ptgaze包安装下来（pip install ptgaze 已经设置了setup文件）
3.输入conda activate pytorch进入创建的虚拟环境pytorch,当发现命令行最左端有(pytorch)，则说明已经成功进入ptorch
4.使用：usage: ptgaze [-h] [--config CONFIG] [--mode {mpiigaze,mpiifacegaze,eth-xgaze}]
              [--face-detector {dlib,face_alignment_dlib,face_alignment_sfd,mediapipe}]
              [--device {cpu,cuda}] [--image IMAGE] [--video VIDEO] [--camera CAMERA]
              [--output-dir OUTPUT_DIR] [--ext {avi,mp4}] [--no-screen] [--debug]
人脸检测器{dlib, face_alignment_dlib、face_alignment_sfd mediapipe}
用于检测人脸和查找人脸地标的方法(默认:
“mediapipe”)
——device {cpu,cuda}用于模型推断的设备。
——image image输入映像文件的路径。
——video video输入视频文件的路径。

在处理图像或视频时，按下窗口上的下列键
显示或隐藏中间结果:
-“l”:地标
-“h”:头部姿势
-“t”:3D人脸模型的投影点
- "b":脸包围框