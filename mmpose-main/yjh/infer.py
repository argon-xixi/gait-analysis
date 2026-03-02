# Check Pytorch installation

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/yjh/code_yjh/segment-anything-main')
sys.path.append('/home/yjh/code_yjh/mmpose-main')
import torch, torchvision
from tqdm import tqdm
print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose
import math
print('mmpose version:', mmpose.__version__)
import cv2
# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())
import matplotlib.pyplot as plt
import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmpose.apis import MMPoseInferencer
from mmpose.utils import adapt_mmdet_pipeline
#API调用
from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

from mmdet.apis import inference_detector, init_detector
from segment_anything import sam_model_registry, SamPredictor
from skimage import morphology
from zs import zhang_suen_thinning
from yjh.find_endpoints import find_endpoints_in_skeleton
from yjh.length_and_width import length_and_width
from pic2vid import getpic,getvid,sort_by_number
from demo.topdown_demo_with_mmdet_new import mmpose
from yjh.footpoint import process_footpoint
from yjh.footpoint_new import  process_frame_footpoint
from yjh.angle import find_angle
from yjh.visual import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 
from scipy.interpolate import CubicSpline 
import time

def ankle_Dorsiflexion(left_HS,right_HS,coords,path_list):
    left_angles=[]
    right_angles=[]
    for i in left_HS:
        # image=cv2.imread(predpath1+path_list[i])
        point_a=(coords[i,2,0],max(coords[:,6,1]))
        point_b=coords[i,6,:]
        point_c=(point_a[0]+2,point_a[1])
        left_angle=find_angle(point_c,point_b,point_a)
        left_angles.append(left_angle)
        # cv2.circle(image, (int(point_a[0]), int(point_a[1])), 4, (0, 0, 255), -1)
        # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/test.jpg',image)
        # print('haha')
    for t in right_HS:
        point_a=(coords[t,5,0],max(coords[:,7,1]))
        point_b=coords[t,7,:]
        point_c=(point_a[0]+2,point_a[1])
        right_angle=find_angle(point_c,point_b,point_a)
        right_angles.append(right_angle)
        
    return left_angle,right_angle
        
        

def getresult(hip_point, knee_point, ankle_point, toe_point,Thigh_length,Thigh_width,calf_length,calf_width):
    perp_point=(hip_point[0],hip_point[1]+2)
    ankle_angle = find_angle(toe_point, knee_point, ankle_point)
    knee_angle = find_angle(hip_point, ankle_point, knee_point)
    hip_angle = find_angle(knee_point, perp_point, hip_point)
    hip_y=hip_point[1]
    knee_y=knee_point[1]
    ankle_y=ankle_point[1]
    return [ankle_angle,knee_angle,hip_angle,hip_y,knee_y,ankle_y,Thigh_length,Thigh_width,calf_length,calf_width]

def getcycletime(time_cycle):
    if len(time_cycle) <2:
        print('未录制到完整周期，请再次录制')
        return 0
    elif len(time_cycle) <4 and len(time_cycle) > 1:
        return time_cycle[-1]-time_cycle[0]
    else:
        return time_cycle[-2]-time_cycle[1]
    
def stance_and_swing(TO,HS,time_cycle):
    key=time_cycle[-1]
    HS_new = [x for x in HS if x > key]
    if not HS_new:
        key=time_cycle[-2]
        HS_new = [x for x in HS if x > key]
    nearest_TO = min(TO, key=lambda x: abs(x - key))
    nearest_HS = min(HS_new, key=lambda x: x - key)
    return nearest_TO,nearest_HS
    # for i in HS:
    #     if i > TO[-2] and i < TO[-1] :
    #         return i-TO[-2] , TO[-1] -i,i
        
start_time=time.time()
videoname='walk_woman_processed_1'
picpath='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/pic'
maskpath='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/mask/'
skeletonpath='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/skeleton/'
predpath='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/pred/'
chartpath='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/'
predpath1='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/pred1/'
# getpic('/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'.mp4','/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/pic/')
if not os.path.exists(maskpath):
    os.makedirs(maskpath)
if not os.path.exists(skeletonpath):
    os.makedirs(skeletonpath)
if not os.path.exists(picpath):
    os.makedirs(picpath)
if not os.path.exists(predpath):
    os.makedirs(predpath)
if not os.path.exists(chartpath):
    os.makedirs(chartpath)
if not os.path.exists(predpath1):
    os.makedirs(predpath1)

for i in os.listdir(picpath):
    img_path=os.path.join(picpath,i)
results=np.zeros((len(os.listdir(picpath)),2,10))
coords= np.zeros((len(os.listdir(picpath)),8,2))
# bboxes=np.zeros((len(os.listdir(picpath)),4,2))

sam_checkpoint = '/home/yjh/code_yjh/sam_vit_h_4b8939.pth' # 预训练模型地址
model_type = "vit_h"

device = "cuda" # 使用GPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 调用预测模型
predictor = SamPredictor(sam)
detect=mmpose(args_input={'det_config': '/home/yjh/code_yjh/mmpose-main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py','det_checkpoint':'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
                            ,'pose_config':'/home/yjh/code_yjh/mmpose-main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py','pose_checkpoint':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
                            ,'draw_heatmap':False,'skeleton_style':'mmpose'})
# angle_data = AngleData()
# chart = AngleChart()
cnt=0
path_list=sorted(os.listdir(picpath), key=sort_by_number)
# 存储上一帧以及最后一次非重叠帧的状态信息
# We store state information from the previous frame and the last non-overlapping frame.
previous_frame_data = {
    'toe_left': None,
    'knee_left': None,
    'ank_left': None,
    'toe_right': None,
    'knee_right': None,
    'ank_right': None,
    
    # 最后一次有效计算出的相对角度 (足膝连线 vs 踝膝连线)
    # The last validly calculated relative angle (toe-knee line vs. ankle-knee line)
    'last_known_relative_angle_left': 0,
    'last_known_relative_angle_right': 0,

    # 相对角度的角速度
    # Angular velocity of the relative angle
    'angular_velocity_left': 0,
    'angular_velocity_right': 0,

    # 最后一次有效计算出的足尖到膝盖的距离
    # The last validly calculated distance from toe to knee
    'toe_knee_radius_left': 0,
    'toe_knee_radius_right': 0
}

for i in path_list:
    cnt+=1
    print(cnt)
    idx=int(i.split('.')[0])
    img_path=os.path.join(picpath,i)
    image = cv2.imread(img_path)
    # 还原原图像色彩
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    device = 'cuda'
    "此处-应该改为_"
    pred_instances,bbox=detect.run(img_path,predpath+i)
    #髋关节11,12，膝关节13,14，踝关节15,16
    #表示各个关节的坐标
    hip_left,hip_right= pred_instances.keypoints[0][11],pred_instances.keypoints[0][12]
    knee_left,knee_right=pred_instances.keypoints[0][13],pred_instances.keypoints[0][14]
    ank_left,ank_right=pred_instances.keypoints[0][15],pred_instances.keypoints[0][16]
    
    # bboxes[idx-1]=bbox
    predictor.set_image(image)
    masks, score, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=False,
    )
    masks=masks[0]
    masks_uint8 = masks.astype(np.uint8) * 255 

    #闭运算
    morph_kernel_size=5
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    processed_mask = cv2.morphologyEx(masks_uint8, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(maskpath+i,processed_mask)
    Thigh_length_left,Thigh_width_left=length_and_width(image,processed_mask,hip_left,knee_left)
    Thigh_length_right,Thigh_width_right=length_and_width(image,processed_mask,hip_right,knee_right)
    calf_length_left,calf_width_left=length_and_width(image,processed_mask,knee_left,ank_left)
    calf_length_right,calf_width_right=length_and_width(image,processed_mask,knee_right,ank_right)
    toe_left, toe_right,previous_frame_data=process_frame_footpoint(processed_mask, ank_left, knee_left, calf_length_left, ank_right, knee_right, calf_length_right,previous_frame_data)
    
    # toe_left=findfootpoint(processed_mask,ank_left,knee_left,1.2*calf_length_left)
    # toe_right=findfootpoint(processed_mask,ank_right,knee_right,1.2*calf_length_right)
    
    image=cv2.imread(predpath+i)
    cv2.circle(image, (int(toe_left[0]), int(toe_left[1])), 4, (0, 0, 255), -1)
    cv2.circle(image, (int(toe_right[0]), int(toe_right[1])), 4, (0, 0, 255), -1)
    cv2.line(image, (int(toe_left[0]), int(toe_left[1])), (int(ank_left[0]), int(ank_left[1])), (85, 255, 0), 2)
    cv2.line(image, (int(toe_right[0]), int(toe_right[1])), (int(ank_right[0]), int(ank_right[1])), (0, 120, 255), 2)
    cv2.imwrite(predpath1+i,image)
    
    coords[idx-1]=[hip_left,knee_left,ank_left,hip_right,knee_right,ank_right,toe_left,toe_right]
    
    result_left=getresult(hip_left, knee_left, ank_left, toe_left, Thigh_length_left, Thigh_width_left, calf_length_left, calf_width_left)
    result_right=getresult(hip_right, knee_right, ank_right, toe_right, Thigh_length_right, Thigh_width_right, calf_length_right, calf_width_right)
    
    results[idx-1]=np.array([result_left,result_right])
  
np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/'+videoname+'_coords.npy',coords)
np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/'+videoname+'_results.npy',results)
# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_bboxes.npy',bboxes)
# x_hip_left,y_hip_left=hip_left[0],hip_left[1]
    # x_hip_right,y_hip_right=hip_right[0],hip_right[1]
    # x_knee_left,y_knee_left=knee_left[0],knee_left[1]
    # x_knee_right,y_knee_right=knee_right[0],knee_right[1]
    # x_ank_left,y_ank_left=ank_left[0],ank_left[1]
    # x_ank_right,y_ank_right=ank_right[0],ank_right[1]
    # Thigh_length_left=math.sqrt((x_hip_left-x_knee_left)**2+(y_hip_left-y_knee_left)**2)
    # Thigh_length_right=math.sqrt((x_hip_right-x_knee_right)**2+(y_hip_right-y_knee_right)**2)
    # calf_length_left=math.sqrt((x_knee_left-x_ank_left)**2+(y_knee_left-y_ank_left)**2)
    # calf_length_right=math.sqrt((x_knee_right-x_ank_right)**2+(y_knee_right-y_ank_right)**2)
    # Thigh_middle_left=((x_hip_left+x_knee_left)/2,(y_hip_left+y_knee_left)/2)
    # Thigh_middle_right=((x_hip_right+x_knee_right)/2,(y_hip_right+y_knee_right)/2)
    # calf_middle_left=((x_knee_left+x_ank_left)/2,(y_knee_left+y_ank_left)/2)
    # calf_middle_right=((x_knee_right+x_ank_right)/2,(y_knee_right+y_ank_right)/2)

  

  

    # dect=mmdetection()
    # input_box = pred_instances.bboxes[0]
    # 通过调用`SamPredictor.set_image`来处理图像以产生一个图像嵌入。`SamPredictor`会记住这个嵌入，并将其用于随后的掩码预测。
    # input_box=np.array([  8.077637, 285.1402 ,1155.2648 ,737.1334 ])
    # input_box=np.array([  13.510618 , 4.175497 ,157.63347 , 221.33658 ])
# bboxes=np.load('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_bboxes.npy')

# coords=np.load('/home/yjh/code_yjh/mmpose-main/yjh/result/'+videoname+'_coords.npy')

print('Thigh_length_left:',max(results[:,0,6]))
print('Thigh_width_left:',np.mean(results[:,0,7]))
print('Thigh_length_right:',max(results[:,1,6]))
print('Thigh_width_right:',np.mean(results[:,1,7]))
print('calf_length_left:',max(results[:,0,8]))
print('calf_width_left:',np.mean(results[:,0,9]))
print('calf_length_right:',max(results[:,1,8]))
print('calf_width_right:',np.mean(results[:,1,9]))


    # angle_data.add_data(result_left[2])
    # chart.update(angle_data)
    
    # cv2.imwrite(chartpath+i,chart.chart_img)


# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_coords.npy',coords)
# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2.npy',results)




# 判断HS和TO以及步态周期
t=np.arange(0,coords.shape[0],1)
left_TO=[]
left_HS=[]
right_TO=[]
right_HS=[]
time_cycle1=[]
time_cycle2=[]
left_angles=[]
right_angles=[]
v_body=(coords[:,0,0]+coords[:,3,0])/2
# body_smooth = savgol_filter(v_body_mean, 40, 4)
# body_cs = CubicSpline(t,body_smooth)  
# body_velocity_spline = body_cs.derivative()  # 速度函数
# v_body = body_velocity_spline(t)
data_raw_x=np.zeros((6,coords.shape[0]))
data_smooth_x=np.zeros((6,coords.shape[0]))
v_x=np.zeros((6,coords.shape[0]))
v_y=np.zeros((6,coords.shape[0]))
# 求出x轴速度
for j in range(6):
    img_path=os.path.join(picpath,path_list[j])
    image = cv2.imread(img_path)
    # 还原原图像色彩
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    data = coords[:,j,0]-v_body
    data_smooth = savgol_filter(data, 40, 4)
    # data = coords[:,j,0]
    # data_smooth = savgol_filter(coords[:,j,0], 40, 4)
    cs = CubicSpline(t,data_smooth)  
    velocity_spline = cs.derivative()  # 速度函数
    v = velocity_spline(t)  # 计算时间点上的速度

    data_raw_x[j]=data
    data_smooth_x[j]=data_smooth 
    v_x[j]=v  
    
# # 左脚
# for m in range(0,v_x.shape[1]-2):
#     if v_x[2,m]-v_x[1,m]<=0 and v_x[2,m+1]-v_x[1,m+1]>0:
#         left_TO.append(m)
#     elif v_x[2,m]-v_x[1,m]>=0 and v_x[2,m+1]-v_x[1,m+1]<0:
#         left_HS.append(m)
# # 右脚
# for m in range(0,v_x.shape[1]-2):
#     if v_x[5,m]-v_x[4,m]<=0 and v_x[5,m+1]-v_x[4,m+1]>0:
#         right_TO.append(m)
#     elif v_x[5,m]-v_x[4,m]>=0 and v_x[5,m+1]-v_x[4,m+1]<0:
#         right_HS.append(m) 
for m in range(1,v_x.shape[1]):
    # 左脚
    key_left=data_smooth_x[2,:]-data_smooth_x[1,:]
    if key_left[m]==min(key_left[max(0,m-5):min(m+5,key_left.shape[0])]):
        if   m < v_x.shape[1]-1:
            left_TO.append(m)
    elif key_left[m]==max(key_left[max(0,m-5):min(m+5,key_left.shape[0])]):
        if  m < v_x.shape[1]-1:
            left_HS.append(m)
    # 右脚
    key_right=data_smooth_x[5,:]-data_smooth_x[4,:]
    if key_right[m]==min(key_right[max(0,m-5):min(m+5,key_right.shape[0])]):
        if  m < v_x.shape[1]-1:
            right_TO.append(m)
    elif key_right[m]==max(key_right[max(0,m-5):min(m+5,key_right.shape[0])]):
        if  m < v_x.shape[1]-1:
            right_HS.append(m) 
    # 步行周期(取往前迈步作为判断标准)
    key1=data_smooth_x[1,:]-data_smooth_x[0,:]
    key2=data_smooth_x[4,:]-data_smooth_x[3,:]
     # 左脚
    if max(key1[max(0,m-5):m]) < 0 and min(key1[m:min(m+5,key1.shape[0])]) > 0:
        if  m < v_x.shape[1]-1:
            time_cycle1.append(m)
    # if max(key1[max(0,m-5):m]) < 0 and min(key1[m:min(m+5,key1.shape[0])]) > 0:
    #     if  m < v_x.shape[1]-1:
    #         time_cycle2.append(m)
    # 右脚
    if max(key2[max(0,m-5):m]) < 0 and min(key2[m:min(m+5,key1.shape[0])]) > 0:
        if  m < v_x.shape[1]-1:
            time_cycle2.append(m)
    # if max(key2[max(0,m-5):m]) < 0 and min(key2[m:min(m+5,key1.shape[0])]) > 0:
    #     if  m < v_x.shape[1]-1:
    #         time_cycle2.append(m)        
left_angles,right_angles=ankle_Dorsiflexion(left_HS,right_HS,coords,path_list)
print(left_angles,right_angles)

# if getcycletime(time_cycle1)*getcycletime(time_cycle2) !=0:
#     time_cycle=(getcycletime(time_cycle1)+getcycletime(time_cycle2))/2
# elif getcycletime(time_cycle1)*getcycletime(time_cycle2) ==0
if len(time_cycle1) >1 and len(time_cycle2) >1:
    time_cycle=(getcycletime(time_cycle1)+getcycletime(time_cycle2))/2
elif len(time_cycle1) >1 and len(time_cycle2) <=1:
    time_cycle=getcycletime(time_cycle1)
elif len(time_cycle1) <=1 and len(time_cycle2) >1:
    time_cycle=getcycletime(time_cycle2)
else:
    print("未录制到完整周期，请重新录制")
TO_left_instance,HS_left_instance=stance_and_swing(left_TO,left_HS,time_cycle1)
TO_right_instance,HS_right_instance=stance_and_swing(right_TO,right_HS,time_cycle2)
swing_left=(HS_left_instance-TO_left_instance)/time_cycle*100
# print('left_TO:',left_TO)
# print('left_HS:',left_HS)
# print('right_TO:',right_TO)
# print('right_HS:',right_HS) 
print('完整周期用时:{}帧' .format(time_cycle))
print('左脚摇摆相占比:{:.3f}%'.format(swing_left))
print('左脚支撑相占比:{:.3f}%'.format(100-swing_left))
print('双支撑相占比:{:.3f}%'.format(min(abs(TO_left_instance-HS_right_instance),abs(TO_right_instance-HS_left_instance))/time_cycle*100))
print('总摇摆相占比:%')

'''#绘结果图
title_list_angle=['ankle_left','ankle_right','knee_left','knee_right','hip_left','hip_right']
title_list_x=['hip_left','knee_left','ank_left','hip_right','knee_right','ank_right']
for idx in range(1,len(results)+1):
    print(idx)
    
    i =  str(idx).zfill(5) + ".jpg"
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    
    # 3行2列
    fig.subplots_adjust(
        left=0.05,      # 左边距
        right=0.95,    # 右边距
        bottom=0.05,    # 底边距
        top=0.95,       # 顶边距
        wspace=0.2,    # 水平子图间距
        hspace=0.3     # 垂直子图间距
    )

    # 为每个子图添加数据并设置坐标轴
    for k, ax in enumerate(axes.flatten()):
        if  k % 2 == 0  :
            data = results[:idx-1,0,k//2]
            data_smooth = savgol_filter(results[:,0,k//2], 40, 4)  
            ax.set_ylim(results[:,0,k//2 ].min()-15,results[:,0,k//2 ].max()+15)
        else:
            data = results[:idx-1,1,k//2 ]
            data_smooth = savgol_filter(results[:,1,k//2], 40, 4)  
            ax.set_ylim(results[:,1,k//2 ].min()-15,results[:,1,k//2 ].max()+15)
        #savgol滤波
        

        
        ax.plot(data,color='blue',label='Raw Signal')
        ax.plot(data_smooth[:idx-1],color='red',label='Smoothed Signal')
        # ax.set_ylim(0, 255)
        ax.set_xlim(0, len(results))
        ax.grid(True)
        ax.legend(loc='upper right',fontsize=6)
        ax.tick_params(labelsize=6)
       
        
        # 设置子图标题（避免重叠关键）
        ax.set_title(title_list_angle[k], fontsize=8,pad=5)  # 减小标题字号
        
        # 仅边缘子图显示标签
        if k not in [0, 3]:
            ax.set_ylabel('')  # 隐藏内部Y轴标签
        if k < 3:
            ax.set_xlabel('')
    plt.savefig('/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/'+i)
    plt.clf()
    plt.close()'''
title_list_angle = ['ankle_left', 'ankle_right', 'knee_left', 'knee_right', 'hip_left', 'hip_right']
title_list_x=['hip_left','knee_left','ank_left','hip_right','knee_right','ank_right']
output_folder = f'/home/yjh/code_yjh/mmpose-main/yjh/ori/{videoname}/chart/'

os.makedirs(output_folder, exist_ok=True)

# 1. 预计算所有平滑数据和Y轴范围
print("Pre-calculating data...")
smoothed_data = np.zeros_like(results)
y_limits = []
for k in range(6):
    col = k % 2
    joint_idx = k // 2
    
    # 计算平滑数据
    # 注意：窗口长度不能大于数据长度，这里假设 len(results) >= 40
    if len(results) >= 40:
        smoothed_data[:, col, joint_idx] = savgol_filter(results[:, col, joint_idx], 40, 4)
    else:
        # 如果数据点太少，无法使用savgol_filter，则直接使用原始数据
        smoothed_data[:, col, joint_idx] = results[:, col, joint_idx]

    # 计算Y轴范围
    min_val = results[:, col, joint_idx].min()
    max_val = results[:, col, joint_idx].max()
    y_limits.append((min_val - 15, max_val + 15))
print("Pre-calculation finished.")

# 2. 创建一个静态的图形和坐标轴
print("Creating static plot canvas...")
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
axes = axes.flatten() # 将2D数组转换为1D，方便索引

# 用于存储所有曲线对象的列表
lines = []

for k, ax in enumerate(axes):
    # 设置所有不动的元素：标题、范围、网格、图例等
    ax.set_ylim(y_limits[k])
    ax.set_xlim(0, len(results))
    ax.grid(True)
    ax.tick_params(labelsize=6)
    ax.set_title(title_list_angle[k], fontsize=8, pad=5)
    
    # 仅边缘子图显示标签
    if k not in [0, 2, 4]:
        ax.set_ylabel('')
    if k < 4: # 原代码是 k<3，但3x2布局中第4个子图(索引为3)也不应该有x轴标签
        ax.set_xlabel('')

    # 创建空的曲线对象，并设置好颜色和标签
    # 我们之后只更新这些对象的数据
    raw_line, = ax.plot([], [], color='blue', label='Raw Signal')
    smooth_line, = ax.plot([], [], color='red', label='Smoothed Signal')
    lines.append((raw_line, smooth_line))
    
    ax.legend(loc='upper right', fontsize=6)
print("Canvas created.")

# 3. 主循环：只更新数据并保存
print("Starting frame generation...")
total_frames = len(results)
x_axis_data = np.arange(total_frames)

for idx in range(1, total_frames + 1):
    if idx % 50 == 0: # 每处理50帧打印一次进度
        print(f"Processing frame {idx}/{total_frames}")

    # 为每个子图更新数据
    for k, (raw_line, smooth_line) in enumerate(lines):
        col = k % 2
        joint_idx = k // 2
        
        # 更新原始数据曲线
        raw_line.set_data(x_axis_data[:idx], results[:idx, col, joint_idx])
        
        # 更新平滑数据曲线
        smooth_line.set_data(x_axis_data[:idx], smoothed_data[:idx, col, joint_idx])

    # 保存整个图形
    filename = str(idx).zfill(5) + ".jpg"
    plt.savefig(os.path.join(output_folder, filename), dpi=100) # 可以适当调整dpi来平衡清晰度和文件大小

# 循环结束后关闭图形，释放内存
plt.close(fig)
print("All frames generated.")

#转成视频
path1='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/pred1/'
path2='/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/'
videopath='/home/yjh/code_yjh/mmpose-main/yjh/result/'+videoname+'.mp4'
getvid(path1,path2,videopath)
end_time=time.time()
print('运行时间：',end_time-start_time)

# # 膝和踝坐标
# # 左脚
# for m in range(1,data_smooth_x.shape[1]-1):
#     if data_smooth_x[2,m-1]-data_smooth_x[1,m-1]>=data_smooth_x[2,m]-data_smooth_x[1,m] and data_smooth_x[2,m+1]-data_smooth_x[1,m+1]>=data_smooth_x[2,m]-data_smooth_x[1,m]:
#         left_TO.append(m)
#     elif data_smooth_x[2,m-1]-data_smooth_x[1,m-1]<=data_smooth_x[2,m]-data_smooth_x[1,m] and data_smooth_x[2,m+1]-data_smooth_x[1,m+1]<=data_smooth_x[2,m]-data_smooth_x[1,m]:
#         left_HS.append(m)
 
# # 右脚       
# for m in range(1,data_smooth_x.shape[1]-1):
#     if data_smooth_x[3,m-1]-data_smooth_x[2,m-1]>=data_smooth_x[3,m]-data_smooth_x[2,m] and data_smooth_x[3,m+1]-data_smooth_x[2,m+1]>=data_smooth_x[3,m]-data_smooth_x[2,m]:
#         right_TO.append(m)
#     elif data_smooth_x[3,m-1]-data_smooth_x[2,m-1]<=data_smooth_x[3,m]-data_smooth_x[2,m] and data_smooth_x[3,m+1]-data_smooth_x[2,m+1]<=data_smooth_x[3,m]-data_smooth_x[2,m]:
#         right_HS.append(m)           

# # 根据y轴速度判断
# for j in range(coords.shape[1]):
#     data = coords[:,j,1]
#     data_smooth = savgol_filter(data, 30, 4)
#     # data = coords[:,j,0]
#     # data_smooth = savgol_filter(coords[:,j,0], 40, 4)
#     cs = CubicSpline(t,data_smooth)  
#     velocity_spline = cs.derivative()  # 速度函数
#     v = velocity_spline(t)  # 计算时间点上的速度
#     if j==2 :
        
#         for m in range(0,len(v)-2):
#             if v[m]<=0 and v[m+1]>0:
#                 left_TO.append(m+1)
#             elif v[m]>=0 and v[m+1]<0:
#                 left_HS.append(m+1)
#     if j==5 :
        
#         for m in range(0,len(v)-2):
#             if v[m]<=0 and v[m+1]>0:
#                 right_TO.append(m)
#             elif v[m]>=0 and v[m+1]<0:
#                 right_HS.append(m)   
#     data_raw_x[j]=data
#     data_smooth_x[j]=data_smooth 
#     v_x[j]=v  
      
# # 根据x轴最大值判断
# for j in range(coords.shape[1]):
#     data = coords[:,j,1]
#     data_smooth = savgol_filter(data, 40, 4)
#     # data = coords[:,j,0]
#     # data_smooth = savgol_filter(coords[:,j,0], 40, 4)
#     # cs = CubicSpline(t,data_smooth)  
#     # velocity_spline = cs.derivative()  # 速度函数
#     # v = velocity_spline(t)  # 计算时间点上的速度
    
#     if j==2 :
        
#         for m in range(1,len(data_smooth)-1):
#             if data_smooth[m-1]>=data_smooth[m] and data_smooth[m+1]>=data_smooth[m]:
#                 left_TO.append(m)
#             elif data_smooth[m-1]<=data_smooth[m] and data_smooth[m+1]<= data_smooth[m] :
#                 left_HS.append(m)
#     if j==5 :
        
#         for m in range(1,len(data_smooth)-1):
#             if data_smooth[m-1]>=data_smooth[m] and data_smooth[m+1]>=data_smooth[m]:
#                 right_TO.append(m)
#             elif data_smooth[m-1]<=data_smooth[m] and data_smooth[m+1]<= data_smooth[m] :
#                 right_HS.append(m)   
#     data_raw_x[j]=data
#     data_smooth_x[j]=data_smooth 
    # v_x[j]=v      


# for idx in tqdm(range(1,len(coords)+1)):
#     print(idx)
    
#     i =  str(idx).zfill(5) + ".jpg"
    
#     fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    
#     # 3行2列
#     fig.subplots_adjust(
#         left=0.05,      # 左边距
#         right=0.95,    # 右边距
#         bottom=0.05,    # 底边距
#         top=0.95,       # 顶边距
#         wspace=0.2,    # 水平子图间距
#         hspace=0.3     # 垂直子图间距
#     )

#     # 为每个子图添加数据并设置坐标轴
#     for k, ax in enumerate(axes.flatten()):
       
        
        
                
            
        
#         # ax.set_ylim(data_raw_x[k].min()-15,data_raw_x[k].max()+15)
#         # ax.set_ylim(data_smooth_x[k].min()-15,data_smooth_x[k].max()+15)
#         ax.set_ylim(v_x[k].min()-15,v_x[k].max()+15)

        
        

        
#         # ax.plot(data_raw_x[k,:idx-1],color='blue',label='Raw Signal')
#         # ax.plot(data_smooth[:idx-1],color='red',label='Smoothed Signal')
#         ax.plot(v_x[k,:idx-1],color='red',label='v_x')
#         # ax.set_ylim(0, 255)
#         ax.set_xlim(0, len(coords))
#         ax.grid(True)
#         ax.legend(loc='upper right',fontsize=6)
#         ax.tick_params(labelsize=6)
       
        
#         # 设置子图标题（避免重叠关键）
#         ax.set_title(title_list_y[k], fontsize=8,pad=5)  # 减小标题字号
    # plt.figure(figsize=(4,3),facecolor='white')
    
    # plt.subplot(3, 2, 1)
    # plt.plot(results[:idx-1,0,0],color='blue')  #第二维度0是左，1是右
    # plt.title('ankle_left')
    # plt.xlim(0, 255)
    # plt.subplot(3, 2, 2)
    # plt.plot(results[:idx-1,0,1],color='blue')
    # plt.title('knee_left')
    # plt.subplot(3, 2, 3)
    # plt.plot(results[:idx-1,0,2],color='blue')
    # plt.title('hip_left')
    # plt.subplot(3, 2, 4)
    # plt.plot(results[:idx-1,1,0],color='blue')
    # plt.title('toe_left')
    # plt.subplot(3, 2, 5)
    # plt.plot(results[:idx-1,1,1],color='blue')
    # plt.title('ankle_right')
    # plt.subplot(3, 2, 6)
    # plt.plot(results[:idx-1,1,2],color='blue')
    # plt.title('knee_right')
    
    

   
    # plt.savefig('/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/knee_left.png')
# for idx in range(1,len(results)+1):
#     i =  str(idx).zfill(5) + ".jpg"

    
    
    
    
#     # 如果没有视频文件，使用摄像头
#     # video_path = 0

#     # angle_data.add_data(results[idx-1,0,2])
        
    
   
#     plt.plot(results[:,0,2],color='blue')
#     plt.scatter(idx,results[idx-1,0,2],marker = '.',c='red')
#     plt.savefig(chartpath+i)
#     plt.clf()
#     print('hhahah')
    # cv2.imwrite(chartpath+i,chart.chart_img)
    
     
    # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1.png',image)
    # global_toe = (toe_point[0] + crop_box[0], toe_point[1] + crop_box[1])
    # skeleton = morphology.skeletonize(processed_mask/255)
    # skeleton=(skeleton * 255).astype(np.uint8)
    # cv2.imwrite(skeletonpath+i,skeleton)

    # Thigh_length_left,Thigh_width_left=length_and_width(image,processed_mask,hip_left,knee_left)
    # Thigh_length_right,Thigh_width_right=length_and_width(image,processed_mask,hip_right,knee_right)
    # calf_length_left,calf_width_left=length_and_width(image,processed_mask,knee_left,ank_left)
    # calf_length_right,calf_width_right=length_and_width(image,processed_mask,knee_right,ank_right)
    # image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(maskpath+i,processed_mask)






# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_mask_new.png',image)


# contours, hierarchy = cv2.findContours(processed_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # contour_image=cv2.drawContours(image, contours, -1, (0,0,255), thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# # # image_final=show_mask(masks[0])
# # contour_image=cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
# # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_contour.png',contour_image)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_mask.png',processed_mask)

# skeleton = zhang_suen_thinning(processed_mask/255)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png',skeleton*255)

# skeleton0 = morphology.skeletonize(processed_mask/255)
# skeleton0=(skeleton0 * 255).astype(np.uint8)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/2_skeleton.png',skeleton0)
# # 查找骨架端点
# endpoints, branches = find_endpoints_in_skeleton(skeleton0)
# print(endpoints)
# # if endpoints:
# #         for point in endpoints:
# #             # y,x = point #cv2需要传入的是x,y,但find_endpoints_in_skeleton 函数返回的坐标是 (y, x) 顺序（NumPy数组索引顺序）
# #             cv2.circle( skeleton0, point, 3, (255, 0, 0), 3)
# footpoint_right=(857, 386)
# footpoint_left=(1152, 559)
# cv2.circle( image, footpoint_left, 3, (255, 0, 0), 3)
# cv2.circle( image, footpoint_right, 3, (255, 0, 0), 3)
# cv2.line(image, (round(ank_left[0]), round(ank_left[1])),footpoint_left, (255, 255, 0), 1)
# cv2.line(image, (round(ank_right[0]), round(ank_right[1])),footpoint_right, (255, 255, 0), 1)

    
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_final.png',image)
     
    

# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png',skeleton0*255)
# img1=cv2.imread('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png')
# contour_image=cv2.drawContours(img1, contours, -1, (0,0,255), thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton_new.png',contour_image)

# for contour in contours:
#     if len(contour) < 3:  # 至少需要3个点
#         continue
    
#     # 找到梯度最大点
#     max_grad_point = find_max_gradient_point(contour)
    
# print(max_grad_point)
# point_image=cv2.circle(processed_mask,max_grad_point, 7, (0, 0, 255), -1)
    # # 可视化 (可选)
    # vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(vis, max_grad_point, 7, (0, 0, 255), -1)  # 红色标记最大梯度点
    # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)  # 绿色绘制轮廓

# img_path = '/home/yjh/code_yjh/mmpose-main/tests/data/coco/000000196141.jpg'   # 将img_path替换给你自己的路径

# # 使用模型别名创建推理器
# # inferencer = MMPoseInferencer('human')

# # # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
# # result_generator = inferencer(img_path, show=False,out_dir='/home/yjh/code_yjh/mmpose-main/yjh')
# # inferencer = MMPoseInferencer('human')

# # 使用模型配置名构建推理器
# # inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')


# # 使用模型配置文件和权重文件的路径或 URL 构建推理器
# inferencer = MMPoseInferencer(
#     pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
#            'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
#     pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
#                    'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
# )
# result_generator = inferencer(img_path, pred_out_dir='predictions')
# result = next(result_generator)


'''python demo/inferencer_demo.py \
/home/yjh/code_yjh/mmpose-main/yjh/final.mp4 \
--pose2d human\
--vis-out-dir vis_results/posetrack18'''

'''

python demo/inferencer_demo.py /home/yjh/code_yjh/mmpose-main/yjh/final.mp4 \
    --pose3d human3d --vis-out-dir vis_results/human3d'''

'''
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input /home/yjh/code_yjh/mmpose-main/yjh/ori/walk_woman_processed_2/pic/00121.jpg \
    --output-root=vis_results  --draw-heatmap'''
    
'''
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/wuyapeng/Downloads/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth  --input /home/yjh/code_yjh/mmpose-main/yjh/final.mp4 --output-root /home/yjh/code_yjh/mmpose-main/vis_results
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/wuyapeng/Downloads/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth  --input webcam --output-root output --show #（自带摄像头）
 '''
 
'''
 python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input /home/yjh/code_yjh/mmpose-main/yjh/1.png  --draw-heatmap --draw-bbox \
    --output-root vis_results/
    '''
