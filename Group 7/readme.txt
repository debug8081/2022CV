组长：王以恒
组员：吴悠 袁忻泽 陈云龙 董子萱
代码运行说明如下：
对于可视化模块，在Visualization文件夹下一共有4个.py文件，分别是chinese.py/identity_processviedo.py/identity_videointopicture.py以及identity_Visualization.py
直接打开identity_Visualization.py，在if __name__ =="__main__":下中输入以下代码：
        video_path = 'F:/human identity/video/4.mp4'视频输入文件路径（包含文件名）
        pic_path = 'F:/human identity/pic/'（视频图片输入路径，不含文件名，下属文件夹就行）
        path = 'F:/human identity/video/'（视频输出文件路径，不含文件名，下属文件夹就行）
        generate_video(video_path)  #无返回值，生成out-4.mp4视频文件
        lmlist = video_into_picture(video_path, pic_path, 1)  #按照1帧将out-4.mp4转换成图片，存储在pic_path中，有返回值，返回一个列表
        vis = Visualization()
        vis.write_csv(lmlist) #将返回的lmlist列表生成一个csv文件，取名为landmark.csv
        frame = vis.count_frames(video_path) #统计视频总帧数，返回值为总帧数
        lmlist_zui = vis.readcsv_get_landmarks('landmark.csv', 9) #获取嘴关键点的纵坐标，返回值为坐标集合列表
        lmlist_jian = vis.readcsv_get_landmarks('landmark.csv', 11) #获取肩关键点的纵坐标，返回值为坐标集合列表
        lmlist_zhou = vis.readcsv_get_landmarks('landmark.csv', 13) #获取肘关键点的纵坐标，返回值为坐标集合列表
        lmlist_shou = vis.readcsv_get_landmarks('landmark.csv', 19) #获取手关键点的纵坐标，返回值为坐标集合列表
        vis.get_plt_picture(frame, lmlist_zui) #无返回值，将嘴部关键点纵坐标进行绘图，横坐标为帧数，纵坐标为嘴部的坐标，并将图片储存在plt_pic中
        vis.paste_picture(frame) #无返回值，利用图片合并，将pic文件夹下的图片与plt_pic下的图片进行整合，合并为一个图片，存储在plt+pic文件夹下
        vis.picture_into_video('output', frame) #无返回值，将plt_pic文件夹下所有的图片转换为一个视频，输出为output.mp4
        vis.put_number_into_video(path + 'output.mp4', frame, lmlist_shou, lmlist_jian, lmlist_zhou, lmlist_zui) #无返回值，将output.mp4视频进行实时计数
        print("已完成")
 下面是可视化的步骤：
        # 1、先将原视频文件进行函数generate_video（），将原视频变成含有33个关键点的视频
        # 2、再将含有33个关键点的视频进行函数video_into_picture()，将视频逐帧处理为图片，并存入pic文件夹中
        # 3、由于函数video_into_picture()这个函数会返回一个带有33个关键点坐标的lmlist列表变量，我们接下来使用write_csv将lmlist列表写入为一个csv文件
        # 4、将生成的csv文件通过函数readcsv_get_landmarks()，获取某个具体的关键点的坐标集合（在视频流下），这会返回一个关键点纵坐标的列表
        # 5、通过使用get_plt_picture()函数读取某一个关键点的坐标集合，生成逐帧坐标图片，无返回值，并储存在plt_pic中
        # 6、再使用paste_picture（）函数将坐标集合进行绘图，横坐标是帧数，纵坐标是关键点的纵坐标，将所有图片储存，储存在plt+pic中
        # 7、再使用picture_into_video（）函数将plt+pic文件下的图片进行转换，储存为视频
        # 8、将最终得到的视频进行put_number_into_video（）进行计数，并实时显示在视频上
对于实时摄像头的运动姿态检测分析，在ActionCount-main中直接运行main.py即可，然后根据ui界面进行操作即可。
