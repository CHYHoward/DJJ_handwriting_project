#1
#standard_path = '1/database/base_1_6_1.bmp'
#test_path = '1/database/base_1_11_1.bmp'

#3
# standard_path = '3/database/base_1_47_3.bmp'
# test_path = '3/database/base_1_7_3.bmp'
#4
# standard_path = '4/database/base_1_11_4.bmp'
# test_path = '4/database/base_1_14_4.bmp'
#5
# standard_path = '5/database/base_1_9_5.bmp'
# test_path = '5/database/base_1_10_5.bmp'
#6

#7
# standard_path = '7/database/base_1_1_7.bmp'
# test_path = '7/database/base_1_11_7.bmp'
#8
# standard_path = '8/database/base_1_2_8.bmp'
# test_path = '8/database/base_1_1_8.bmp'
#9
# standard_path = '9/database/base_1_2_9.bmp'
# test_path = '9/database/base_1_6_9.bmp'


standard_path = '1/database/base_1_47_1.bmp'
test_path = '1/database/base_1_11_1.bmp'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
# 读取图像
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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

    # # 打印每个轮廓的点座标和角度
    # for idx, contour in enumerate(all_contours):
    #     print(f"Contour {idx+1}:")
    #     for point_idx, point in enumerate(contour):
    #         theta_n = calculate_angle(contour, point_idx)
    #         print(f"{point_idx + 1}: ({point[1]}, {point[0]}), θ_{point_idx + 1}: {theta_n:.2f}°")
        
    # print(f"Total number of contours with 20 or more points: {len(all_contours)}")


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


    #-------------------0819--------------------------------

    # 示例数据，实际使用时替换为你的数据
    # stroke_coords = np.array([[y1, x1], [y2, x2], ...])
    # corner_points = [[y1, x1], [y2, x2], ...]

    # 找出最左、最右、最上、最下的点
    min_x = min(stroke_coords[:, 1])
    max_x = max(stroke_coords[:, 1])
    min_y = min(stroke_coords[:, 0])
    max_y = max(stroke_coords[:, 0])

    mc = ((max_y - min_y) / 2) + min_y
    nc = ((max_x - min_x) / 2) + min_x
    L = max(mc, nc)

    # print(f"mc: {mc}, nc: {nc}, L: {L}")

    # 存储角点的所有属性
    corner_data = []

    for idx, corner in enumerate(corner_points):
        # 计算 normalized_x 和 normalized_y
        normalized_x = round(((corner[1] - nc) / L) * 100, 2)
        normalized_y = round(((corner[0] - mc) / L) * 100, 2)
        
        # 计算 rx 和 ry
        rx = round(corner[1] / 189, 2)
        ry = round(corner[0] / 189, 2)
        
        # 存储角点数据  
        # corner_data.append({
        #     'index': idx + 1,
        #     'normalized_x': normalized_x,
        #     'normalized_y': normalized_y,
        #     'rx': rx,
        #     'ry': ry
        # })  # 存储角点属性到 corner_data list中  # 注意：此处使用了 dict 作为存储的数据类型，可以自行修改为其他数据类型，如 list、tuple、numpy array 等
        corner_data.append(corner[1])
        corner_data.append(corner[0])
        corner_data.append(normalized_x)
        corner_data.append(normalized_y)
        corner_data.append(rx)
        corner_data.append(ry)
        
        # 画出角点
        # cv2.circle(image, (int(corner[1]), int(corner[0])), 2, (0, 0, 255), -1)  # 画出红色圆点
        # cv2.putText(image, f"Corner {idx+1}", (int(corner[1]), int(corner[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0
        
        # print(f"Corner {idx + 1}: normalized_x={normalized_x}, normalized_y={normalized_y}, rx={rx}, ry={ry}")

    # print("Corner Data:")
    # for data in corner_data:
    #     print(data)

    return corner_data

standard_data = []
test_data = []

standard_data = process_image(standard_path)
test_data = process_image(test_path)
print(standard_data)
print(test_data)

standard_corner_num = len(standard_data) // 6
test_corner_num = len(test_data) // 6

print(test_corner_num)
print(standard_corner_num)
# print (standard_data[2])

# 創一個list存放兩兩對應的corner的座標
for_picture = []

for i in range(standard_corner_num):
    
    # distance[i] =  ((standard_normalized_x - test_normalized_x)^2 + (standard_normalized_y - test_normalized_y)^2 + (standard_rx - test_rx)^2 + (standard_ry - test_ry)^2 + )^0.5
    min_distance = math.inf
    for j in range(test_corner_num):
        distance = (((standard_data[i*6+2] - test_data[j*6+2])**2 + (standard_data[i*6+3] - test_data[j*6+3])**2 + 10000*(standard_data[i*6+4] - test_data[j*6+4])**2 + 10000*(standard_data[i*6+5] - test_data[j*6+5])**2)**0.5)
        if distance < min_distance:
            min_distance = distance
            min_index = j
    print(f"The minimum distance between corner {i+1} and corner {min_index+1} is {min_distance:.2f}")
    #印出對應的角點座標
    print(f"Standard corner {i+1} is ({standard_data[i*6]},{standard_data[i*6+1]})")
    print(f"Test corner {min_index+1} is ({test_data[min_index*6]},{test_data[min_index*6+1]})")
    for_picture.append(standard_data[i*6])
    for_picture.append(standard_data[i*6+1])
    for_picture.append(test_data[min_index*6])
    for_picture.append(test_data[min_index*6+1])
    
import matplotlib.pyplot as plt
from PIL import Image

# 加載標準圖和測試圖
standard_img = Image.open(standard_path)  # 替換為你的標準圖路徑
test_img = Image.open(test_path)  # 替換為你的測試圖路徑

# 提供的座標數據，格式為 [標準圖 x1, 標準圖 y1, 測試圖 x1, 測試圖 y1, ...]
coordinates = for_picture
# 檢查座標數據是否為4的倍數（每組點需要4個數據）
if len(coordinates) % 4 != 0:
    raise ValueError("座標數據長度必須是4的倍數，每組點應包含4個數據（標準圖x, 標準圖y, 測試圖x, 測試圖y）")

# 提取標準圖和測試圖的角點座標
standard_corners = [(coordinates[i*4], coordinates[i*4+1]) for i in range(len(coordinates)//4)]
test_corners = [(coordinates[i*4+2], coordinates[i*4+3]) for i in range(len(coordinates)//4)]

# 定義顏色列表，每對點使用相同顏色
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']  # 根據需要添加更多顏色

# 畫標準圖和標註角點
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(standard_img)
for i, (x, y) in enumerate(standard_corners):
    plt.scatter(x, y, c=colors[i % len(colors)], marker='o')  # 使用相應顏色
    plt.text(x, y, f'{i+1}', color='black', fontsize=12, ha='right')
plt.title("Standard Image with Corners")

# 畫測試圖和標註角點
plt.subplot(1, 2, 2)
plt.imshow(test_img)
for i, (x, y) in enumerate(test_corners):
    plt.scatter(x, y, c=colors[i % len(colors)], marker='o')  # 使用相應顏色
    plt.text(x, y, f'{i+1}', color='black', fontsize=12, ha='right')
plt.title("Test Image with Corners")

plt.show()


