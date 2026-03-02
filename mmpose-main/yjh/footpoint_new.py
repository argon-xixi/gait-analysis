import os
import cv2
import numpy as np


def draw_rect(image, point_ankle, point_knee, length):
    """根据脚踝和膝盖点绘制一个矩形ROI。"""
    # (此函数与之前版本相同，无需修改)
    x_a, y_a = point_ankle
    x_b, y_b = point_knee
    dx = x_b - x_a
    dy = y_b - y_a
    perp_dx = -dy
    perp_dy = dx
    norm_perp = np.sqrt(perp_dx**2 + perp_dy**2)
    if norm_perp == 0: return np.zeros(image.shape, dtype=np.uint8)
    scale_perp = length / (2 * norm_perp)
    x1, y1 = (x_a + scale_perp * perp_dx, y_a + scale_perp * perp_dy)
    x2, y2 = (x_a - scale_perp * perp_dx, y_a - scale_perp * perp_dy)
    para_dx, para_dy = -dx, -dy
    norm_para = np.sqrt(para_dx**2 + para_dy**2)
    if norm_para == 0: return np.zeros(image.shape, dtype=np.uint8)
    scale_para = 0.8 * length / (2 * norm_para)
    x3, y3 = (round(x1 + scale_para * para_dx), round(y1 + scale_para * para_dy))
    x4, y4 = (round(x2 + scale_para * para_dx), round(y2 + scale_para * para_dy))
    img_bg = np.zeros(image.shape, dtype=np.uint8)
    img_rect = cv2.fillConvexPoly(img_bg, np.array([[x1, y1], [x3, y3], [x4, y4], [x2, y2]], dtype=np.int32), 1)
    return img_rect

def findfootpoint(processed_mask, point_ankle, point_knee, length):
    """在指定的矩形ROI内通过PCA找到足尖。"""
    # (此函数与之前版本相同，无需修改)
    img_rect = draw_rect(processed_mask, point_ankle, point_knee, length)
    masked_area = processed_mask * img_rect
    if np.sum(masked_area) == 0: return (int(point_ankle[0]), int(point_ankle[1]))
    contours, _ = cv2.findContours(masked_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return (int(point_ankle[0]), int(point_ankle[1]))
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        M = cv2.moments(largest_contour)
        if M["m00"] == 0: return (int(point_ankle[0]), int(point_ankle[1]))
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    largest_contour_float = largest_contour.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(largest_contour_float.reshape(-1, 2), mean=None)
    direction = eigenvectors[0]
    projected = np.dot(largest_contour_float.reshape(-1, 2) - mean.flatten(), direction)
    toe_index = np.argmax(projected)
    toe_point = largest_contour[toe_index][0]
    return (int(toe_point[0]), int(toe_point[1]))

def calculate_angle_between_vectors(vec1, vec2):
    """计算从 vec1 到 vec2 的有符号角度。"""
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])
    angle = angle2 - angle1
    # 标准化到 [-pi, pi]
    if angle > np.pi: angle -= 2 * np.pi
    elif angle < -np.pi: angle += 2 * np.pi
    return angle

def predict_toe_with_relative_rotation(current_knee, current_ankle, radius, last_relative_angle, angular_velocity):
    """
    使用相对角速度预测新的足尖位置。
    """
    # 1. 计算当前参考向量 (踝-膝) 的角度
    #    Calculate the angle of the current reference vector (ankle-to-knee).
    vec_ank_knee_current = np.array(current_ankle) - np.array(current_knee)
    ref_angle = np.arctan2(vec_ank_knee_current[1], vec_ank_knee_current[0])

    # 2. 预测新的相对角度
    #    Predict the new relative angle.
    new_relative_angle = last_relative_angle + angular_velocity

    # 3. 计算足尖向量的绝对角度
    #    Calculate the absolute angle of the toe vector.
    #    这里的逻辑是：足尖向量的角度 = 参考向量(踝膝)的角度 + 相对角度
    #    The logic here: angle_of_toe_vector = angle_of_ref_vector + relative_angle
    #    注意：我们这里要计算的是 toe-knee 向量，其相对于 ankle-knee 向量的角度是 new_relative_angle
    #    The angle of (toe-knee) relative to (ankle-knee) is `new_relative_angle`.
    #    So, the absolute angle of (toe-knee) is `ref_angle + new_relative_angle`.
    absolute_toe_angle = ref_angle + new_relative_angle
    
    # 4. 根据半径和新角度计算新足尖位置
    #    Calculate the new toe position based on the radius and new angle.
    new_toe_x = current_knee[0] + radius * np.cos(absolute_toe_angle)
    new_toe_y = current_knee[1] + radius * np.sin(absolute_toe_angle)
    
    return (int(new_toe_x), int(new_toe_y))

# --- 主处理循环 ---
def process_frame_footpoint(processed_mask, ank_left, knee_left, calf_length_left, ank_right, knee_right, calf_length_right,previous_frame_data):


    rect_left = draw_rect(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)
    rect_right = draw_rect(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)

    overlap_area = cv2.bitwise_and(rect_left, rect_right)
    is_overlapping = np.sum(overlap_area) > 0

    if is_overlapping:
        print("检测到重叠区域，使用相对角速度模型进行预测。")
        
        # --- 左脚预测 ---
        # 仅当所有必要历史数据都存在时才进行预测
        if all(previous_frame_data[k] is not None for k in previous_frame_data.keys()):
            if  previous_frame_data['ank_left'][1] < previous_frame_data['ank_right'][1]:
                toe_left = predict_toe_with_relative_rotation(
                    knee_left,
                    ank_left,
                    previous_frame_data['toe_knee_radius_left'],
                    previous_frame_data['last_known_relative_angle_left'],
                    previous_frame_data['angular_velocity_left']
                )
            else:
                # 历史 toe 点平移：使用本帧踝点相对上一帧踝点的位移近似 toe 位移。
                # 注意：toe_left 是 tuple，直接与 numpy array 相加会类型报错，这里统一转 np.array。
                toe_left = tuple(
                    (np.array(previous_frame_data['toe_left'])
                     + (np.array(ank_left) - np.array(previous_frame_data['ank_left']))
                     ).astype(int)
                )
        else: # 退化情况：如果缺少历史数据，则仍使用PCA
            toe_left = findfootpoint(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)
            vec_ank_knee_curr = np.array(ank_left) - np.array(knee_left)
            vec_toe_knee_curr = np.array(toe_left) - np.array(knee_left)
            current_relative_angle = calculate_angle_between_vectors(vec_ank_knee_curr, vec_toe_knee_curr)
            previous_frame_data['last_known_relative_angle_left'] = current_relative_angle
            previous_frame_data['toe_knee_radius_left'] = np.linalg.norm(np.array(toe_left) - np.array(knee_left))

        # --- 右脚预测 ---
        if all(previous_frame_data[k] is not None for k in previous_frame_data.keys()):
            if  previous_frame_data['ank_left'][1] > previous_frame_data['ank_right'][1]:
                toe_right = predict_toe_with_relative_rotation(
                    knee_right,
                    ank_right,
                    previous_frame_data['toe_knee_radius_right'],
                    previous_frame_data['last_known_relative_angle_right'],
                    previous_frame_data['angular_velocity_right']
                )
            else:
                toe_right = tuple(
                    (np.array(previous_frame_data['toe_right'])
                     + (np.array(ank_right) - np.array(previous_frame_data['ank_right']))
                     ).astype(int)
                )
                vec_ank_knee_curr = np.array(ank_right) - np.array(knee_right)
                vec_toe_knee_curr = np.array(toe_right) - np.array(knee_right)
                current_relative_angle = calculate_angle_between_vectors(vec_ank_knee_curr, vec_toe_knee_curr)
                previous_frame_data['last_known_relative_angle_right'] = current_relative_angle
                previous_frame_data['toe_knee_radius_right'] = np.linalg.norm(np.array(toe_right) - np.array(knee_right))
        else: # 退化情况
            toe_right = findfootpoint(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)
        

    else:
        print("无重叠,使用PCA方法并更新运动状态。")
        # 如果不重叠，使用原来的方法找到足尖
        toe_left = findfootpoint(processed_mask, ank_left, knee_left, 1.2 * calf_length_left)
        toe_right = findfootpoint(processed_mask, ank_right, knee_right, 1.2 * calf_length_right)

        # 并且，更新所有运动状态变量
        # And, update all motion state variables.
        
        # --- 左脚状态更新 ---
        if previous_frame_data['knee_left'] is not None and previous_frame_data['ank_left'] is not None : # 需要有上一帧才能计算角速度
            # 1. 计算当前帧的相对角度
            vec_ank_knee_curr = np.array(ank_left) - np.array(knee_left)
            vec_toe_knee_curr = np.array(toe_left) - np.array(knee_left)
            current_relative_angle = calculate_angle_between_vectors(vec_ank_knee_curr, vec_toe_knee_curr)
            
            # 2. 计算角速度 (当前相对角度 - 上一次的相对角度)
            angular_velocity = current_relative_angle - previous_frame_data['last_known_relative_angle_left']
            previous_frame_data['angular_velocity_left'] = angular_velocity
            
            # 3. 更新“最后已知的”相对角度
            previous_frame_data['last_known_relative_angle_left'] = current_relative_angle

        # 4. 更新半径 (每次非重叠时都更新)
        previous_frame_data['toe_knee_radius_left'] = np.linalg.norm(np.array(toe_left) - np.array(knee_left))

        # --- 右脚状态更新 ---
        if previous_frame_data['knee_right'] is not None and previous_frame_data['ank_right'] is not None:
            vec_ank_knee_curr = np.array(ank_right) - np.array(knee_right)
            vec_toe_knee_curr = np.array(toe_right) - np.array(knee_right)
            current_relative_angle = calculate_angle_between_vectors(vec_ank_knee_curr, vec_toe_knee_curr)
            angular_velocity = current_relative_angle - previous_frame_data['last_known_relative_angle_right']
            previous_frame_data['angular_velocity_right'] = angular_velocity
            previous_frame_data['last_known_relative_angle_right'] = current_relative_angle
        previous_frame_data['toe_knee_radius_right'] = np.linalg.norm(np.array(toe_right) - np.array(knee_right))


    # 更新上一帧的关节点数据以备下一帧使用
    # Update last frame's joint data for the next frame.
    previous_frame_data['toe_left'] = toe_left
    previous_frame_data['knee_left'] = knee_left
    previous_frame_data['ank_left'] = ank_left
    previous_frame_data['toe_right'] = toe_right
    previous_frame_data['knee_right'] = knee_right
    previous_frame_data['ank_right'] = ank_right
    

    return toe_left, toe_right,previous_frame_data