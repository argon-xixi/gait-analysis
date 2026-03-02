import cv2
import numpy as np
from yjh.find_endpoints import find_endpoints_in_skeleton


def find_min_abs_two_per_column(arr,j=1):
    """
    
    
    参数:
        arr: 二维数组
        
    返回:
        min_abs_values: 形状为 (2, cols) 的数组，包含每列绝对值最小的两个值
        min_abs_indices: 形状为 (2, cols) 的数组，包含这些值的行索引
    """
    # 计算绝对值数组
    abs_arr = np.abs(arr)
    rows, cols = arr.shape
    
    # 为结果预分配空间
    
    
    
    
    
    # 获取当前列
    col = abs_arr[:, j]
    
    # 使用 argpartition 高效找到最小的两个索引
    # 注意：kth=1 表示第二个最小值的索引
    min_indices = np.argpartition(col, kth=1)[:2]
    
   
   
    
    
    return  min_indices

def draw_perpendicular_at_midpoint(image, point_a, point_b, length=10000, color=(1, 1, 1), thickness=1):
    """
    在AB中点绘制垂直线
    
    参数:
        image: 要绘制的图像
        point_a: 点A的坐标 (x_a, y_a)
        point_b: 点B的坐标 (x_b, y_b)
        length: 垂直线长度（像素）
        color: 线条颜色 (B, G, R)
        thickness: 线条粗细
    """
    # 解包坐标
    x_a, y_a = point_a
    x_b, y_b = point_b
    
    # 1. 计算中点P
    x_p = (x_a + x_b) / 2
    y_p = (y_a + y_b) / 2
    midpoint = (int(round(x_p)), int(round(y_p)))
    
    # 2. 计算AB方向向量
    dx = x_b - x_a
    dy = y_b - y_a
    
    # 3. 计算垂直向量（旋转90度）
    # 有两种可能方向：(-dy, dx) 或 (dy, -dx)
    # 这里选择第一个方向 (-dy, dx)
    perp_dx = -dy
    perp_dy = dx
    
    # 4. 标准化垂直向量并缩放到指定长度
    norm = np.sqrt(perp_dx**2 + perp_dy**2)
    if norm < 1e-10:  # 避免除以零（A和B重合）
        return image
    
    scale = length / (2 * norm)  # 一半长度，因为我们要向两边延伸
    
    # 5. 计算垂直线两个端点
    # 从P点向两个方向延伸
    x1 = x_p + scale * perp_dx
    y1 = y_p + scale * perp_dy
    x2 = x_p - scale * perp_dx
    y2 = y_p - scale * perp_dy
    
    # 转换为整数坐标
    pt1 = (int(round(x1)), int(round(y1)))
    pt2 = (int(round(x2)), int(round(y2)))
    
    # 6. 绘制垂直线
    cv2.line(image, pt1, pt2, color, thickness)
    
    # # 7. 标记中点
    # cv2.circle(image, midpoint, 5, (0, 0, 255), -1)  # 红色
    
    # # 8. 可选：绘制原始AB线段
    # cv2.line(image, 
    #          (int(round(x_a)), int(round(y_a))),
    #          (int(round(x_b)), int(round(y_b))),
    #          (0, 255, 255), 1)  # 黄色
    
    return image[:,:,0],midpoint

def length_and_width(image,mask, point_a, point_b):
    # 计算线段长度
    length = np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
    # 计算线段宽度
    width = np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)
    img_bg= np.zeros(image.shape, dtype=np.uint8)

    result_img,mid_points = draw_perpendicular_at_midpoint( img_bg, point_a, point_b)
    endpoints, branches=find_endpoints_in_skeleton(mask/255*result_img)
    
    endpoints_temp=np.array(endpoints)-np.array(mid_points)
    indices=find_min_abs_two_per_column(endpoints_temp)
    point_c=endpoints[indices[0]]
    point_d=endpoints[indices[1]]
    width= np.sqrt((point_c[0] - point_d[0])**2 + (point_c[1] - point_d[1])**2)
    cv2.line(image, point_c, point_d, (255, 255, 255), 1)
    return length,width
    
    