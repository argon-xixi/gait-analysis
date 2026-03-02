import cv2
import os 
import numpy as np
import re

def sort_by_number(filename):
    # 使用正则表达式匹配文件名中的数字部分
    match = re.search(r'(\d+)', filename)
    if match:
        # 将匹配的数字转换为整数用于排序
        return int(match.group(1))
    else:
        # 如果没有数字，返回一个很大的数，确保这些文件排在最后
        return float('inf')
    
def getpic(videoPath, svPath):#两个参数，视频源地址和图片保存地址
    cap = cv2.VideoCapture(videoPath)

    numFrame = 0
    while True:
        # 函数cv2.VideoCapture.grab()用来指向下一帧，其语法格式为：
        # 如果该函数成功指向下一帧，则返回值retval为True
        if cap.grab():
            # 函数cv2.VideoCapture.retrieve()用来解码，并返回函数cv2.VideoCapture.grab()捕获的视频帧。该函数的语法格式为：
            # retval, image = cv2.VideoCapture.retrieve()image为返回的视频帧，如果未成功，则返回一个空图像。retval为布尔类型，若未成功，返回False；否则返回True
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                numFrame += 1
                #设置图片存储路径
                newPath = svPath + str(numFrame).zfill(5) + ".jpg"
                #注意这块利用.zfill函数是为了填充图片名字，即如果不加这个，那么图片排序后大概就是1.png,10.png,...2.png,这种顺序合成视频就会造成闪烁，因此增加zfill，变为00001.png,00002.png;可以根据生成图片的数量大致调整zfill的值
                # cv2.imencode()函数是将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输。
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
                print(numFrame)
        else:
            break

    print("all is ok")
    
def getvid(path1,path2,videopath):

    img_array = []
    


    
    
    filelist = os.listdir(path1) 
    # 对文件列表进行排序
    sorted_filelist = sorted(filelist, key=sort_by_number)
    # 获取该目录下的所有文件名
    for filename in sorted_filelist:
        #挨个读取图片
        img1 = cv2.imread(path1+filename)
        img2 = cv2.imread(path2+filename)
        resized = cv2.resize(img1, (1280,800))

        # img2 = np.resize(img2,(320,320))
        
        img = cv2.hconcat([resized,img2]) #水平拼接
        # img=img1
        # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/ori/walk_woman_processed_1/chart/knee_left.png',img)
        # img = cv2.vconcat([img1,img2]) #竖直拼接
        #获取图片高，宽，通道数信息
        height, width, layers = img.shape
        #设置尺寸
        size = (width, height)
        #将图片添加到一个大“数组中”
        img_array.append(img)
    print("this is ok")
    # avi：视频类型，mp4也可以
    # cv2.VideoWriter_fourcc(*'DIVX')：编码格式，不同的编码格式有不同的视频存储类型
    # fps：视频帧率
    # size:视频中图片大小
    fps=10
    
    out1 = cv2.VideoWriter(videopath,cv2.VideoWriter_fourcc(*'DIVX'),fps, size)
    for i in range(len(img_array)):
        #写成视频操作
        out1.write(img_array[i])
    out1.release()
    print("all is ok")
