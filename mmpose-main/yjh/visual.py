# import cv2
# import numpy as np
# import time
# from collections import deque

# # 配置参数
# class Config:
#     # 图表参数
#     CHART_WIDTH = 320
#     CHART_HEIGHT = 320
#     CHART_MARGIN = 20
#     CHART_POSITION = (20, 20)  # 左上角位置
#     BACKGROUND_COLOR = (255, 255, 255)  # 深灰色背景
#     GRID_COLOR = (80, 80, 80)  # 网格颜色
#     LINE_COLOR = (0, 200, 255)  # 曲线颜色 (BGR)
#     CURRENT_POINT_COLOR = (0, 0, 255)  # 当前点颜色 (红色)
#     TEXT_COLOR = (200, 200, 200)  # 文本颜色
    
#     # 图表坐标范围
#     TIME_RANGE = 225  # 显示的时间范围 (帧)
#     ANGLE_MIN = 0  # 角度最小值
#     ANGLE_MAX = 90  # 角度最大值
    
#     # 视频处理
#     SHOW_FPS = True
#     MAX_FRAMES = 1000  # 最大处理帧数

# # 角度数据存储类
# class AngleData:
#     def __init__(self):
#         self.timestamps = deque(maxlen=Config.TIME_RANGE)
#         self.angles = deque(maxlen=Config.TIME_RANGE)
#         self.frame_count = 0
#         self.start_time = time.time()
    
#     def add_data(self, angle):
#         """添加新的角度数据"""
#         self.frame_count += 1
#         self.timestamps.append(self.frame_count)
#         self.angles.append(angle)
    
#     def get_current_data(self):
#         """获取当前数据点"""
#         if self.angles:
#             return self.timestamps[-1], self.angles[-1]
#         return None, None

# # 图表绘制类
# class AngleChart:
#     def __init__(self):
#         # 创建空白图表图像
#         self.chart_img = np.zeros((Config.CHART_HEIGHT, Config.CHART_WIDTH, 3), dtype=np.uint8)
#         self.chart_img[:] = Config.BACKGROUND_COLOR
#         self.timestamps = deque(maxlen=Config.TIME_RANGE)
#         # 绘制静态元素
#         self._draw_static_elements()
    
#     def _draw_static_elements(self):
#         """绘制静态元素（网格、坐标轴等）"""
#         # 绘制外框
#         cv2.rectangle(self.chart_img, 
#                      (0, 0), 
#                      (Config.CHART_WIDTH-1, Config.CHART_HEIGHT-1), 
#                      Config.GRID_COLOR, 1)
        
#         # 绘制网格线
#         # 水平网格线（角度）
#         for angle in range(Config.ANGLE_MIN, Config.ANGLE_MAX+1, 15):
#             y = self._angle_to_y(angle)
#             cv2.line(self.chart_img, 
#                     (Config.CHART_MARGIN, y), 
#                     (Config.CHART_WIDTH - Config.CHART_MARGIN, y), 
#                     Config.GRID_COLOR, 1)
#             cv2.putText(self.chart_img, f"{angle}", 
#                        (5, y - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
#                        Config.TEXT_COLOR, 1)
        
#         # 垂直网格线（时间）
#         for i in range(0, Config.TIME_RANGE+1, 30):
#             x = self._time_to_x(i)
#             if Config.CHART_MARGIN <= x < Config.CHART_WIDTH - Config.CHART_MARGIN:
#                 cv2.line(self.chart_img, 
#                         (x, Config.CHART_MARGIN), 
#                         (x, Config.CHART_HEIGHT - Config.CHART_MARGIN), 
#                         Config.GRID_COLOR, 1)
        
#         # 绘制标题
#         cv2.putText(self.chart_img, "Foot Angle Over Time", 
#                    (Config.CHART_WIDTH//2 - 100, 15), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                    Config.TEXT_COLOR, 1)
    
#     def _time_to_x(self, timestamp, max_time=None):
#         """将时间戳转换为图表中的x坐标"""
#         if max_time is None:
#             max_time = max(self.timestamps) if self.timestamps else Config.TIME_RANGE
        
#         min_time = max(0, max_time - Config.TIME_RANGE)
#         rel_time = timestamp - min_time
#         x = Config.CHART_MARGIN + int(rel_time * (Config.CHART_WIDTH - 2*Config.CHART_MARGIN) / Config.TIME_RANGE)
#         return min(max(Config.CHART_MARGIN, x), Config.CHART_WIDTH - Config.CHART_MARGIN - 1)
    
#     def _angle_to_y(self, angle):
#         """将角度转换为图表中的y坐标"""
#         rel_angle = (angle - Config.ANGLE_MIN) / (Config.ANGLE_MAX - Config.ANGLE_MIN)
#         y = Config.CHART_HEIGHT - Config.CHART_MARGIN - int(rel_angle * (Config.CHART_HEIGHT - 2*Config.CHART_MARGIN))
#         return min(max(Config.CHART_MARGIN, y), Config.CHART_HEIGHT - Config.CHART_MARGIN - 1)
    
#     def update(self, data):
#         """更新图表数据"""
#         # 创建新的图表图像（保留静态元素）
#         self.chart_img = np.zeros((Config.CHART_HEIGHT, Config.CHART_WIDTH, 3), dtype=np.uint8)
#         self.chart_img[:] = Config.BACKGROUND_COLOR
#         self._draw_static_elements()
        
#         # 如果没有数据，返回空图表
#         if len(data.angles) < 2:
#             return
        
#         # 绘制角度曲线
#         points = []
#         max_time = data.timestamps[-1]
        
#         for i in range(len(data.timestamps)):
#             x = self._time_to_x(data.timestamps[i], max_time)
#             y = self._angle_to_y(data.angles[i])
#             points.append((x, y))
        
#         # 绘制曲线
#         for i in range(1, len(points)):
#             cv2.line(self.chart_img, points[i-1], points[i], Config.LINE_COLOR, 2)
        
#         # 绘制当前点
#         if data.angles:
#             curr_x = self._time_to_x(data.timestamps[-1], max_time)
#             curr_y = self._angle_to_y(data.angles[-1])
#             cv2.circle(self.chart_img, (curr_x, curr_y), 6, Config.CURRENT_POINT_COLOR, -1)
            
#             # 添加当前值标签
#             label = f"{data.angles[-1]:.1f}"
#             cv2.putText(self.chart_img, label, 
#                        (curr_x + 10, curr_y - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                        Config.CURRENT_POINT_COLOR, 2)
        
#         # 添加帧数标签
#         frame_label = f"Frame: {data.frame_count}"
#         cv2.putText(self.chart_img, frame_label, 
#                    (Config.CHART_WIDTH - 120, Config.CHART_HEIGHT - 10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
#                    Config.TEXT_COLOR, 1)




# 主处理函数
# def process_video(video_path):
#     """处理视频并实时显示角度变化图"""
#     # 初始化
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return
    
#     # 获取视频信息
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # 创建数据存储和图表
#     angle_data = AngleData()
#     chart = AngleChart()
    
#     # 创建窗口
#     cv2.namedWindow("Foot Angle Tracking", cv2.WINDOW_NORMAL)
    
#     # 帧处理循环
#     frame_count = 0
#     prev_time = time.time()
    
#     while cap.isOpened() and frame_count < Config.MAX_FRAMES:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
 
        
#         # 2. 计算角度
#         angle = calculate_foot_angle(ankle_point, toe_point)
#         if angle is not None:
#             angle_data.add_data(angle)
        
#         # 3. 更新图表
#         chart.update(angle_data)
        

        
#         # 将图表叠加到视频帧上
#         chart_x, chart_y = Config.CHART_POSITION
#         frame[chart_y:chart_y+Config.CHART_HEIGHT, 
#               chart_x:chart_x+Config.CHART_WIDTH] = chart.chart_img
        
#         # 显示当前角度值
#         if angle is not None:
#             angle_text = f"Foot Angle: {angle:.1f}°"
#             cv2.putText(frame, angle_text, 
#                        (width - 300, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
#                        (0, 200, 255), 2)
        
#         # 显示FPS
#         if Config.SHOW_FPS:
#             current_time = time.time()
#             fps = 1.0 / (current_time - prev_time)
#             prev_time = current_time
#             cv2.putText(frame, f"FPS: {fps:.1f}", 
#                        (width - 150, height - 20), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                        (0, 255, 0), 2)
        
#         # 显示处理后的帧
#         cv2.imshow("Foot Angle Tracking", frame)
        
#         # 保存帧（可选）
#         # cv2.imwrite(f"frame_{frame_count:04d}.jpg", frame)
        
#         # 按 'q' 退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # 释放资源
#     cap.release()
#     cv2.destroyAllWindows()

# # 主函数
# if __name__ == "__main__":
    # 使用示例视频（替换为您的视频路径）
    
    
    # 如果没有视频文件，使用摄像头
    # video_path = 0
    # angle_data = AngleData()
    # chart = AngleChart()
    
    # angle_data.add_data(angle)
        
    
    # chart.update(angle_data)
            
    # chart_x, chart_y = Config.CHART_POSITION
    # cv2.imwrite(,chart.chart_img)
    
    