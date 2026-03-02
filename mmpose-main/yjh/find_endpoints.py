import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
from collections import defaultdict, deque
import matplotlib.pyplot as plt

def find_endpoints_in_skeleton(skeleton):
    """
    从骨架图像中提取所有折线的端点坐标
    
    参数:
        skeleton: 二值骨架图像 (2D numpy数组), 骨架像素为1, 背景为0
        
    返回:
        endpoints: 所有端点坐标的列表 [(y1, x1), (y2, x2), ...]
        branches: 所有分支的列表, 每个分支是坐标点的列表
    """
    # 标记连通分量
    labeled_skeleton = measure.label(skeleton, connectivity=2)
    regions = measure.regionprops(labeled_skeleton)
    
    endpoints = []
    branches = []
    
    for region in regions:
        # 获取当前连通分量的所有坐标
        coords = region.coords
        
        # 构建邻接图
        graph = defaultdict(list)
        coord_set = set([(y, x) for y, x in coords])
        
        # 建立邻接关系 (8连通)
        for y, x in coords:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in coord_set:
                        graph[(y, x)].append((ny, nx))
        
        # 找出所有端点 (度为1的节点)
        local_endpoints = [point for point, neighbors in graph.items() if len(neighbors) == 1]
        
        # 如果整个分量是一个孤立点 (度为0)
        if len(graph) == 0 and len(coords) == 1:
            local_endpoints = [(coords[0][0], coords[0][1])]
        
        endpoints.extend(local_endpoints)
        
        # 提取所有分支
        visited = set()
        for endpoint in local_endpoints:
            if endpoint in visited:
                continue
                
            # 从端点开始遍历分支
            branch = []
            queue = deque([endpoint])
            visited.add(endpoint)
            
            while queue:
                current = queue.popleft()
                branch.append(current)
                
                # 只添加未访问的邻居
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            branches.append(branch)
    endpoints_xy = []
    for point in endpoints:
        y,x= point
        endpoints_xy.append((x,y))
        
    
    return endpoints_xy, branches
