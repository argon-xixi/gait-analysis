import cv2 
import math

import numpy as np



def predict_toe_with_angular_velocity(prev_toe, prev_knee, current_knee, angular_velocity):
    """
    使用角速度预测新的足尖位置。
    """
    # 计算上一帧中足尖相对于膝盖的向量
    prev_vec = np.array(prev_toe) - np.array(prev_knee)
    
    # 计算上一帧的角度
    prev_angle = np.arctan2(prev_vec[1], prev_vec[0])
    
    # 计算新的角度
    new_angle = prev_angle + angular_velocity
    
    # 计算半径（足尖到膝盖的距离）
    radius = np.linalg.norm(prev_vec)
    
    # 计算新的足尖位置
    new_toe_x = current_knee[0] + radius * np.cos(new_angle)
    new_toe_y = current_knee[1] + radius * np.sin(new_angle)
    
    return (int(new_toe_x), int(new_toe_y))

def calculate_angular_velocity(prev_toe, prev_knee, current_toe, current_knee):
    """
    计算足尖相对于膝盖的角速度。
    """
    vec_prev = np.array(prev_toe) - np.array(prev_knee)
    vec_curr = np.array(current_toe) - np.array(current_knee)
    
    angle_prev = np.arctan2(vec_prev[1], vec_prev[0])
    angle_curr = np.arctan2(vec_curr[1], vec_curr[0])
    
    # 简单的角速度（弧度/帧）
    angular_velocity = angle_curr - angle_prev

    # 处理角度跳变 (e.g., from pi to -pi)
    if angular_velocity > np.pi:
        angular_velocity -= 2 * np.pi
    elif angular_velocity < -np.pi:
        angular_velocity += 2 * np.pi
        
    return angular_velocity

def draw_rect(image,point_ankle, point_knee,length):
    x_a, y_a = point_ankle
    x_b, y_b = point_knee
    #计算ab向量
    dx = x_b - x_a
    dy = y_b - y_a

    
    #计算垂直向量
    perp_dx = -dy
    perp_dy = dx
    norm_perp = np.sqrt(perp_dx**2 + perp_dy**2)
    scale_perp = length / (2 * norm_perp)
    x1 = x_a + scale_perp * perp_dx
    y1 = y_a + scale_perp * perp_dy
    x2 = x_a - scale_perp * perp_dx
    y2 = y_a - scale_perp * perp_dy
    
    #计算平行向量
    para_dx=-dx
    para_dy=-dy

    norm_para = np.sqrt(para_dx**2 + para_dy**2)
    scale_para = 0.8*length / (2 * norm_para)
    x3=round(x1+scale_para*para_dx)
    y3=round(y1+scale_para*para_dy)
    x4=round(x2+scale_para*para_dx)
    y4=round(y2+scale_para*para_dy)
    # x5=round(x1-scale_para*para_dx)
    # y5=round(y1-scale_para*para_dy)
    # x6=round(x2-scale_para*para_dx)
    # y6=round(y2-scale_para*para_dy)
    img_bg= np.zeros(image.shape, dtype=np.uint8)
    img_rect=cv2.fillConvexPoly(img_bg, np.array([[x1,y1],[x3,y3],[x4,y4],[x2,y2]], dtype=np.int32), 1)
    # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/rect.png',img_rect*image)
    
    
    return img_rect

# def findfootpoint(processed_mask,point_ankle,point_knee,length):
#     img_rect = draw_rect(processed_mask,point_ankle, point_knee,length)
#     contours, hierarchy = cv2.findContours(processed_mask*img_rect,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     largest_contour = np.array(max(contours, key=cv2.contourArea)).astype(np.float32)
#     mean, eigenvectors = cv2.PCACompute(largest_contour.reshape(-1,2), mean=None)
#     direction = eigenvectors[0] 
#     projected = np.dot(largest_contour - mean, direction)
#     toe_index = np.argmax(projected)
#     toe_point = largest_contour[toe_index][0]# 主方向向量
#     footpoint = (int(toe_point[0]), int(toe_point[1]))
#     # toe=cv2.circle(image, footpoint ,3, (255, 0, 0), 3)
#     # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/toe.png',toe)
#     return footpoint

def findfootpoint(processed_mask, point_ankle, point_knee, length):
    """
    在指定的矩形ROI内通过PCA找到足尖。
    """
    img_rect = draw_rect(processed_mask, point_ankle, point_knee, length)
    masked_area = processed_mask * img_rect
    
    # 确保有轮廓可以寻找
    if np.sum(masked_area) == 0:
        # 如果没有分割区域，可以返回一个默认值或者脚踝的位置
        return (int(point_ankle[0]), int(point_ankle[1]))
        
    contours, _ = cv2.findContours(masked_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (int(point_ankle[0]), int(point_ankle[1]))

    largest_contour = max(contours, key=cv2.contourArea)
    
    # 确保轮廓点数足够进行PCA
    if len(largest_contour) < 5:
        # 如果轮廓点太少，PCA可能不稳定，直接返回轮廓的质心
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return (int(point_ankle[0]), int(point_ankle[1]))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    largest_contour_float = largest_contour.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(largest_contour_float.reshape(-1, 2), mean=None)
    direction = eigenvectors[0]
    projected = np.dot(largest_contour_float.reshape(-1, 2) - mean.flatten(), direction)
    toe_index = np.argmax(projected)
    toe_point = largest_contour[toe_index][0]
    footpoint = (int(toe_point[0]), int(toe_point[1]))
    return footpoint

def calculate_angular_velocity(prev_toe, prev_knee, current_toe, current_knee):
    """
    计算足尖相对于膝盖的角速度。
    """
    vec_prev = np.array(prev_toe) - np.array(prev_knee)
    vec_curr = np.array(current_toe) - np.array(current_knee)
    
    angle_prev = np.arctan2(vec_prev[1], vec_prev[0])
    angle_curr = np.arctan2(vec_curr[1], vec_curr[0])
    
    # 简单的角速度（弧度/帧）
    angular_velocity = angle_curr - angle_prev

    # 处理角度跳变 (e.g., from pi to -pi)
    if angular_velocity > np.pi:
        angular_velocity -= 2 * np.pi
    elif angular_velocity < -np.pi:
        angular_velocity += 2 * np.pi
        
    return angular_velocity

def process_footpoint(processed_mask, ank_left, knee_left, calf_length_left, ank_right, knee_right, calf_length_right,previous_frame_data):


    # 1. 绘制左右脚的矩形ROI
    rect_left = draw_rect(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)
    rect_right = draw_rect(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)

    # 2. 检测矩形区域是否重叠
    # 使用位与操作来找到重叠区域
    overlap_area = cv2.bitwise_and(rect_left, rect_right)
    is_overlapping = np.sum(overlap_area) > 0

    if is_overlapping:
        print("检测到重叠区域，使用角速度预测。")
        # 如果重叠，采用新的方式预测足尖坐标
        # 我们需要上一帧的数据来进行预测
        
        # --- 左脚预测 ---
        if previous_frame_data['toe_left'] is not None and previous_frame_data['knee_left'] is not None:
            toe_left = predict_toe_with_angular_velocity(
                previous_frame_data['toe_left'],
                previous_frame_data['knee_left'],
                knee_left,
                previous_frame_data['angular_velocity_left']
            )
        else:
            # 如果没有上一帧数据，退回到原始方法
            toe_left = findfootpoint(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)

        # --- 右脚预测 ---
        if previous_frame_data['toe_right'] is not None and previous_frame_data['knee_right'] is not None:
            toe_right = predict_toe_with_angular_velocity(
                previous_frame_data['toe_right'],
                previous_frame_data['knee_right'],
                knee_right,
                previous_frame_data['angular_velocity_right']
            )
        else:
            # 如果没有上一帧数据，退回到原始方法
            toe_right = findfootpoint(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)

    else:
        # 3. 如果不重叠，使用原来的方法
        print("无重叠,使用PCA方法。")
        toe_left = findfootpoint(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)
        toe_right = findfootpoint(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)

        # 4. 更新角速度 (只有在不重叠且有上一帧数据时才计算)
        if previous_frame_data['toe_left'] is not None and previous_frame_data['knee_left'] is not None:
            previous_frame_data['angular_velocity_left'] = calculate_angular_velocity(
                previous_frame_data['toe_left'], previous_frame_data['knee_left'],
                toe_left, knee_left
            )
        if previous_frame_data['toe_right'] is not None is not None and previous_frame_data['knee_right'] is not None:
             previous_frame_data['angular_velocity_right'] = calculate_angular_velocity(
                previous_frame_data['toe_right'], previous_frame_data['knee_right'],
                toe_right, knee_right
            )

    # 5. 更新上一帧的数据以备下一帧使用
    previous_frame_data['toe_left'] = toe_left
    previous_frame_data['knee_left'] = knee_left
    previous_frame_data['toe_right'] = toe_right
    previous_frame_data['knee_right'] = knee_right

    return toe_left, toe_right,previous_frame_data


if __name__ == '__main__':
    pass


    
    