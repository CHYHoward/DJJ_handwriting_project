

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 读取图像
image = cv2.imread('8/database/base_1_47_8.bmp', cv2.IMREAD_GRAYSCALE)

# 计算强度
intensity = image
stroke_or_background = (intensity < 220).astype(int)
B = stroke_or_background

# 获取所有stroke像素的坐标
stroke_coords = np.column_stack(np.where(B == 1))

# 用于存储边界点坐标
boundary_points = []

for (m, n) in stroke_coords:
    if (m > 0 and B[m-1, n] == 0) or (m < B.shape[0]-1 and B[m+1, n] == 0) or \
       (n > 0 and B[m, n-1] == 0) or (n < B.shape[1]-1 and B[m, n+1] == 0):
        boundary_points.append((m, n))

# 边界点排序，找到最左上角的点
def find_starting_point(points):
    return min(points, key=lambda x: (x[1], -x[0]))

# 根据前一个点的位置确定顺时针方向
def determine_search_order(prev_m, prev_n, cur_m, cur_n):
    delta_m = cur_m - prev_m
    delta_n = cur_n - prev_n
    
    if delta_m == 0 and delta_n == 1:  # 右
        return [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif delta_m == 0 and delta_n == -1:  # 左
        return [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    elif delta_m == -1 and delta_n == 0:  # 上
        return [(1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]
    elif delta_m == 1 and delta_n == 0:  # 下
        return [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
    elif delta_m == -1 and delta_n == 1:  # 右上
        return [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    elif delta_m == -1 and delta_n == -1:  # 左上
        return [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    elif delta_m == 1 and delta_n == -1:  # 左下
        return [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    elif delta_m == 1 and delta_n == 1:  # 右下
        return [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]    
    else:
        return [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

# 顺时针方向检查八个邻域
def find_next_point(B, prev_m, prev_n, cur_m, cur_n, boundary_points):
    directions = determine_search_order(prev_m, prev_n, cur_m, cur_n)
    
    for direction in directions:
        new_m, new_n = cur_m + direction[0], cur_n + direction[1]
        
        if 0 <= new_m < B.shape[0] and 0 <= new_n < B.shape[1] and (new_m, new_n) in boundary_points:
            return new_m, new_n
    return None

# 记录已访问的点
visited = set()

# 找到起始点
start_point = find_starting_point(boundary_points)
current_contour = [start_point]
visited.add(start_point)

cur_m, cur_n = start_point
prev_m, prev_n = cur_m, cur_n - 1

# 记录所有轮廓的列表
all_contours = []
all_points = []

while len(visited) < len(boundary_points):
    while True:
        next_point = find_next_point(B, prev_m, prev_n, cur_m, cur_n, boundary_points)
        if next_point == start_point:
            break
        if next_point is None:
            break
        prev_m, prev_n = cur_m, cur_n
        cur_m, cur_n = next_point
        current_contour.append((cur_m, cur_n))
        visited.add((cur_m, cur_n))
        all_points.append((cur_m, cur_n))
    
    if len(current_contour) >= 20:
        all_contours.append(current_contour)
    
    if len(visited) >= len(boundary_points):
        break
    
    remaining_points = [point for point in boundary_points if point not in visited]
    if remaining_points:
        start_point = find_starting_point(remaining_points)
        current_contour = [start_point]
        visited.add(start_point)
        cur_m, cur_n = start_point
        prev_m, prev_n = cur_m, cur_n - 1

# 计算角度
def calculate_angle(contour, n):
    N = len(contour)
    p1 = contour[(n - 10) % N]
    p2 = contour[n]
    p3 = contour[(n + 10) % N]
    
    v1 = np.array([p1[1] - p2[1], p1[0] - p2[0]])  # 向量 (x1-x2, y1-y2)
    v2 = np.array([p3[1] - p2[1], p3[0] - p2[0]])  # 向量 (x3-x2, y3-y2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_theta = dot_product / (norm_v1 * norm_v2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    return np.degrees(theta)

# 打印每个轮廓的点座标和角度
for idx, contour in enumerate(all_contours):
    print(f"Contour {idx+1}:")
    for point_idx, point in enumerate(contour):
        theta_n = calculate_angle(contour, point_idx)
        print(f"{point_idx + 1}: ({point[1]}, {point[0]}), θ_{point_idx + 1}: {theta_n:.2f}°")
    
print(f"Total number of contours with 20 or more points: {len(all_contours)}")
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to calculate the angle θ between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / magnitude
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)

# Function to find corners in a contour
def find_corners(contour):
    corners = []
    contour_len = len(contour)
    
    for n in range(contour_len):
        prev_idx = (n - 10) % contour_len
        next_idx = (n + 10) % contour_len
        
        v1 = np.array(contour[prev_idx]) - np.array(contour[n])
        v2 = np.array(contour[next_idx]) - np.array(contour[n])
        
        theta_n = calculate_angle(v1, v2)
        
        # Check the surrounding 20 points (10 before and 10 after)
        is_corner = theta_n < 135
        for i in range(1, 11):
            if calculate_angle(np.array(contour[(n - 10 - i) % contour_len]) - np.array(contour[(n - i) % contour_len]), 
                               np.array(contour[(n + 10 - i) % contour_len]) - np.array(contour[(n - i) % contour_len])) <= theta_n:
                is_corner = False
                break
            if calculate_angle(np.array(contour[(n - 10 + i) % contour_len]) - np.array(contour[(n + i) % contour_len]), 
                               np.array(contour[(n + 10 + i) % contour_len]) - np.array(contour[(n + i) % contour_len])) <= theta_n:
                is_corner = False
                break
        
        if is_corner:
            corners.append(contour[n])
    
    return corners

# Assuming 'all_contours' contains all the detected contours
corner_points = []

for contour in all_contours:
    if len(contour) > 20:
        corners = find_corners(contour)
        corner_points.extend(corners)

# Mark the corners on the image with small green dots (pixel-level)
corner_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for point in corner_points:
    corner_image[point[0], point[1]] = (0, 255, 0)  # Set the pixel to green

# Display the image with corners marked
plt.figure(figsize=(10, 10))
plt.imshow(corner_image)
plt.title("Corners Detected")
plt.show()

# Print corner coordinates
for idx, corner in enumerate(corner_points):
    print(f"Corner {idx+1}: ({corner[1]}, {corner[0]})")

