# ---------沒有考慮過去的contour是否有包含新的contour------------


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 读取图像
image = cv2.imread('3/database/base_1_1_3.bmp', cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取圖像

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
    # 找x最大的，且x最大的點中找y最小的
    return min(points, key=lambda x: (x[1], -x[0]))

# 根据前一个点的位置确定顺时针方向
def determine_search_order(prev_m, prev_n, cur_m, cur_n):
    # 计算当前点相对于前一个点的位置
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
prev_m, prev_n = cur_m, cur_n - 1  # 初始上一点设为左边的点

# 记录所有轮廓的列表
all_contours = []
all_points = []

while len(visited) < len(boundary_points):
    while True:
        next_point = find_next_point(B, prev_m, prev_n, cur_m, cur_n, boundary_points)
        if next_point is None or next_point == start_point:
            break
        prev_m, prev_n = cur_m, cur_n
        cur_m, cur_n = next_point
        current_contour.append((cur_m, cur_n))
        visited.add((cur_m, cur_n))
        all_points.append((cur_m, cur_n))
    
    all_contours.append(current_contour)
    
    if len(visited) >= len(boundary_points):
        break
    
    remaining_points = [point for point in boundary_points if point not in visited]
    if remaining_points:
        start_point = find_starting_point(remaining_points)
        current_contour = [start_point]
        visited.add(start_point)
        cur_m, cur_n = start_point
        prev_m, prev_n = cur_m, cur_n - 1  # 初始上一点设为左边的点

# 繪製動畫
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def update(num):
    if num < len(all_points):
        point = all_points[num]
        contour_image[point[0], point[1]] = (0, 0, 255)  # 用紅色標記邊界點
        ax.clear()
        ax.imshow(contour_image)

ani = animation.FuncAnimation(fig, update, frames=len(all_points), interval=0.1, repeat=False)
plt.show()

# 打印每个轮廓的点座标
for idx, contour in enumerate(all_contours):
    print(f"Contour {idx+1}: {contour}")
    
print(f"Total number of contours: {len(all_contours)}")

# ---------有考慮過去的contour是否有包含新的contour------------


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # 读取图像
# image = cv2.imread('2/database/base_1_1_2.bmp', cv2.IMREAD_GRAYSCALE)  # 以灰度模式讀取圖像

# # 计算强度
# intensity = image
# stroke_or_background = (intensity < 220).astype(int)
# B = stroke_or_background

# # 获取所有stroke像素的坐标
# stroke_coords = np.column_stack(np.where(B == 1))

# # 用于存储边界点坐标
# boundary_points = []

# for (m, n) in stroke_coords:
#     if (m > 0 and B[m-1, n] == 0) or (m < B.shape[0]-1 and B[m+1, n] == 0) or \
#        (n > 0 and B[m, n-1] == 0) or (n < B.shape[1]-1 and B[m, n+1] == 0):
#         boundary_points.append((m, n))

# # 边界点排序，找到最左上角的点
# def find_starting_point(points):
#     # 找x最大的，且x最大的點中找y最小的
#     return min(points, key=lambda x: (x[1], -x[0]))

# # 根据前一个点的位置确定顺时针方向
# def determine_search_order(prev_m, prev_n, cur_m, cur_n):
#     # 计算当前点相对于前一个点的位置
#     delta_m = cur_m - prev_m
#     delta_n = cur_n - prev_n
    
#     if delta_m == 0 and delta_n == 1:  # 右
#         return [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
#     elif delta_m == 0 and delta_n == -1:  # 左
#         return [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
#     elif delta_m == -1 and delta_n == 0:  # 上
#         return [(1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]
#     elif delta_m == 1 and delta_n == 0:  # 下
#         return [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
#     elif delta_m == -1 and delta_n == 1:  # 右上
#         return [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
#     elif delta_m == -1 and delta_n == -1:  # 左上
#         return [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
#     elif delta_m == 1 and delta_n == -1:  # 左下
#         return [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
#     elif delta_m == 1 and delta_n == 1:  # 右下
#         return [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]    
#     else:
#         return [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

# # 顺时针方向检查八个邻域
# def find_next_point(B, prev_m, prev_n, cur_m, cur_n, boundary_points, visited):
#     directions = determine_search_order(prev_m, prev_n, cur_m, cur_n)
    
#     for direction in directions:
#         new_m, new_n = cur_m + direction[0], cur_n + direction[1]
        
#         if 0 <= new_m < B.shape[0] and 0 <= new_n < B.shape[1] and (new_m, new_n) in boundary_points and (new_m, new_n) not in visited:
#             return new_m, new_n
#     return None

# # 记录已访问的点
# visited = set()

# # 找到起始点
# start_point = find_starting_point(boundary_points)
# current_contour = [start_point]
# visited.add(start_point)

# cur_m, cur_n = start_point
# prev_m, prev_n = cur_m, cur_n - 1  # 初始上一点设为左边的点

# # 记录所有轮廓的列表
# all_contours = []
# all_points = [start_point]

# while len(visited) < len(boundary_points):
#     while True:
#         next_point = find_next_point(B, prev_m, prev_n, cur_m, cur_n, boundary_points, visited)
#         if next_point is None or next_point == start_point:
#             break
#         prev_m, prev_n = cur_m, cur_n
#         cur_m, cur_n = next_point
#         current_contour.append((cur_m, cur_n))
#         visited.add((cur_m, cur_n))
#         all_points.append((cur_m, cur_n))
    
#     all_contours.append(current_contour)
    
#     if len(visited) >= len(boundary_points):
#         break
    
#     remaining_points = [point for point in boundary_points if point not in visited]
#     if remaining_points:
#         start_point = find_starting_point(remaining_points)
#         current_contour = [start_point]
#         visited.add(start_point)
#         all_points.append(start_point)
#         cur_m, cur_n = start_point
#         prev_m, prev_n = cur_m, cur_n - 1  # 初始上一点设为左边的点

# # 繪製動畫
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(image, cmap='gray')
# contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# def update(num):
#     if num < len(all_points):
#         point = all_points[num]
#         contour_image[point[0], point[1]] = (0, 0, 255)  # 用紅色標記邊界點
#         ax.clear()
#         ax.imshow(contour_image)

# ani = animation.FuncAnimation(fig, update, frames=len(all_points), interval=0.1, repeat=False)
# plt.show()

# # 打印每个轮廓的点座标
# for idx, contour in enumerate(all_contours):
#     print(f"Contour {idx+1}: {contour}")
    
# print(f"Total number of contours: {len(all_contours)}")