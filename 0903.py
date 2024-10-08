from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# 定义文件路径
database_path = '8/database/'
testcase_path = '8/testcase/'

# 读取图像文件列表
database_files = sorted([f for f in os.listdir(database_path) if f.endswith('.bmp')])
testcase_files = sorted([f for f in os.listdir(testcase_path) if f.endswith('.bmp')])

# 随机打乱文件列表
database_files = shuffle(database_files, random_state=30)
testcase_files = shuffle(testcase_files, random_state=30)

# 分割文件列表为训练集和测试集
train_database_files = database_files[:25]
test_database_files = database_files[25:]
train_testcase_files = testcase_files[:25]
test_testcase_files = testcase_files[25:]
#--------------------------------0722 erosion ---------------------------
def erode_image(binary_image):
    eroded_image = np.copy(binary_image)
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            eroded_image[i, j] = binary_image[i, j] & binary_image[i, j+1] & binary_image[i, j-1] & binary_image[i+1, j] & binary_image[i-1, j]
    return eroded_image
#--------------------------------0722 erosion ---------------------------
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
#--------------------------------0903 Corner------------------------------------------------
#--------------------------------0708 basic ten feature ---------------------------

# 提取特征的函数
def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    intensity = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    stroke_or_background = (intensity < 220).astype(int)

    height, width = stroke_or_background.shape
    horizontal_split = width // 5
    vertical_split = height // 5

    features = []

    for i in range(5):
        start_col = i * horizontal_split
        end_col = (i + 1) * horizontal_split if i < 4 else width
        region = stroke_or_background[:, start_col:end_col]
        stroke_count = np.sum(region)
        features.append(stroke_count)

    for i in range(5):
        start_row = i * vertical_split
        end_row = (i + 1) * vertical_split if i < 4 else height
        region = stroke_or_background[start_row:end_row, :]
        stroke_count = np.sum(region)
        features.append(stroke_count)
#--------------------------------0708 basic ten feature ---------------------------
#--------------------------------0722 intensity mean std ---------------------------
    # 計算 stroke pixel 的 intensity 平均值和標準差
    stroke_intensity = intensity[stroke_or_background == 1]
    if len(stroke_intensity) > 0:
        mean_intensity = np.mean(stroke_intensity)
        std_intensity = np.std(stroke_intensity)
    else:
        mean_intensity = 0
        std_intensity = 0

    features.append(mean_intensity)
    features.append(std_intensity)
#--------------------------------0722 intensity mean std -----------------------
#--------------------------------0715 moment feature ---------------------------
    # moment features
    B = stroke_or_background
    M00 = np.sum(B)
    M10 = np.sum(np.arange(width) * np.sum(B, axis=0))
    M01 = np.sum(np.arange(height) * np.sum(B, axis=1))
    
    m0 = M01 / M00
    n0 = M10 / M00
    # print(m0)
    # print(n0)
    height, width = B.shape
    
    m20 = 0
    for i in range(width):
        for j in range(height):
            m20 += ((j - m0) ** 2) * B[i, j]
    m20 = m20/M00
    # print(m20)
    
    m02 = 0
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m02 += ((i - n0) ** 2) * B[i, j]
    m02 = m02/M00
    # print(m02)    

    m11 = 0
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m11 += (i - n0) * (j - m0) * B[i, j]
    m11 = m11/M00
    # print(m11)

    m30 = 0
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m30 += ((j - m0) ** 3) * B[i, j]
    m30 = m30/M00
    # print(m30)

    m21 = 0
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m21 += (i - n0) * ((j - m0) ** 2) * B[i, j]
    m21 = m21/M00
    # print(m21)

    m12 = 0 
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m12 += ((j - m0)) * ((i - n0) **2 ) * B[i, j]
    m12 = m12/M00
    # print(m12)

    m03 = 0
    for i in range(width):
        for j in range(height):
            # if B[i, j] == 1:
            m03 += ((i - n0) ** 3) * B[i, j]
    m03 = m03/M00
    
    features.append(m0)
    features.append(n0)
    features.append(m20)
    features.append(m02)
    features.append(m11)
    features.append(m30)
    features.append(m21)
    features.append(m12)
    features.append(m03)
#--------------------------------0715 moment feature ---------------------------
#--------------------------------0722 erosion ---------------------------
    #erosion
    B1 = erode_image(B)
    B2 = erode_image(B1)
    B3 = erode_image(B2)
    r1 = np.sum(B1)/np.sum(B)
    r2 = np.sum(B2)/np.sum(B)
    r3 = np.sum(B3)/np.sum(B)
    
    # print(f"r0:{M00},r1: {r1}, r2: {r2}, r3: {r3}")
    features.append(r1)
    features.append(r2)
    features.append(r3)
    # 計算��界框的面��
#--------------------------------0722 erosion ---------------------------
#--------------------------------0729 PCA--------------------------------
    # 获取所有stroke像素的坐标
    stroke_coords = np.column_stack(np.where(B == 1))
  
    # 计算所有stroke像素的均值坐标
    mean_x, mean_y = np.mean(stroke_coords, axis=0)
    # print (mean_x, mean_y)
    
    # 生成特征矩阵Z
    Z = stroke_coords - np.array([mean_x, mean_y])
    # 生成Z的轉置矩陣
    Z_transpose = Z.T
    #Z Z_transpose相乘
    
    Z2 = np.dot(Z_transpose, Z)
    # print(Z2)
     # 计算Z_product的特征分解
    eigenvalues, eigenvectors = np.linalg.eig(Z2)
    # 生成对角矩阵D
    D = np.diag(eigenvalues)
    # 特征向量矩阵E
    E = eigenvectors
    
    ref_vector = np.array([1, 0])
    angles = []
    
    for eigenvector in E.T:
        # print(eigenvector  )
        cos_theta = np.dot(eigenvector, ref_vector) / (np.linalg.norm(eigenvector) * np.linalg.norm(ref_vector))
        angle = np.arccos(cos_theta)
        # angle = np.degrees(angle)
        # print(angle_deg)
        angles.append(angle)
    
    # 找到较小夹角对应的特征向量
    #如果較小的angle不在-45度到45度要計算
    min_angle_index = np.argmax(np.abs(angles))
    
    if min_angle_index == 0:
        lambda_horizontal = eigenvalues[0]
        lambda_vertical = eigenvalues[1]
        angle_horizontal = angles[0]
    else:
        lambda_horizontal = eigenvalues[1]
        lambda_vertical = eigenvalues[0]
        angle_horizontal = angles[1]
    
    # 将角度从弧度转换为度
    angle_horizontal_degrees = np.degrees(angle_horizontal)
        # 将角度限制在 -90 到 90 度之间
    if angle_horizontal_degrees > 90:
        angle_horizontal_degrees -= 180
    elif angle_horizontal_degrees < -90:
        angle_horizontal_degrees += 180
    # 將添加到列表中
    features.append(lambda_horizontal)
    features.append(lambda_vertical)
    features.append(angle_horizontal_degrees)
#--------------------------------0729 PCA--------------------------------
#--------------------------------0903 Corner------------------------------------------------
    B = stroke_or_background

    # 获取所有stroke像素的坐标
    stroke_coords = np.column_stack(np.where(B == 1))

    # 用于存储边界点坐标
    boundary_points = []

    for (m, n) in stroke_coords:
        if (m > 0 and B[m-1, n] == 0) or (m < B.shape[0]-1 and B[m+1, n] == 0) or \
        (n > 0 and B[m, n-1] == 0) or (n < B.shape[1]-1 and B[m, n+1] == 0):
            boundary_points.append((m, n))

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


    # # 打印每个轮廓的点座标和角度
    # for idx, contour in enumerate(all_contours):
    #     print(f"Contour {idx+1}:")
    #     for point_idx, point in enumerate(contour):
    #         theta_n = calculate_angle(contour, point_idx)
    #         # print(f"{point_idx + 1}: ({point[1]}, {point[0]}), θ_{point_idx + 1}: {theta_n:.2f}°")
        
    # print(f"Total number of contours with 20 or more points: {len(all_contours)}")


    # Assuming 'all_contours' contains all the detected contours
    corner_points = []
    
    
    
    for contour in all_contours:
        if len(contour) > 20:
            corners = find_corners(contour)
            corner_points.extend(corners)
    #分別印出第五個corner_points的x,y座標
    # # Mark the corners on the image with small green dots (pixel-level)
    # corner_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for point in corner_points:
    #     corner_image[point[0], point[1]] = (0, 255, 0)  # Set the pixel to green

    # # Display the image with corners marked
    # plt.figure(figsize=(10, 10))
    # plt.imshow(corner_image)
    # plt.title("Corners Detected")
    # plt.show()

    # Print corner coordinates
    # for idx, corner in enumerate(corner_points):
    #     # print(f"Corner {idx+1}: ({corner[1]}, {corner[0]})")
    if database_path == '9/database/':
        print("this is test for 9")
        features.append(corner_points[5][1])
        features.append(corner_points[5][0])
        features.append(corner_points[7][1])
        features.append(corner_points[7][0])
        features.append(corner_points[14][1])
        features.append(corner_points[14][0])
        # features.append(corner_points[16][1])
        # features.append(corner_points[16][0])
        features.append(corner_points[5][1] - corner_points[7][1])
        features.append(corner_points[5][0] - corner_points[7][0])
        features.append(corner_points[5][1] - corner_points[14][1])
        features.append(corner_points[5][0] - corner_points[14][0])
        # features.append(corner_points[5][1] - corner_points[16][1])
        # features.append(corner_points[5][0] - corner_points[16][0])
        features.append(corner_points[7][1] - corner_points[14][1])
        features.append(corner_points[7][0] - corner_points[14][0])
        # features.append(corner_points[7][1] - corner_points[16][1])
        # features.append(corner_points[7][0] - corner_points[16][0])
        # features.append(corner_points[14][1] - corner_points[16][1])
        # features.append(corner_points[14][0] - corner_points[16][0])
    elif database_path == '8/database/':
        print("this is test for 8")
        features.append(corner_points[7][1])
        features.append(corner_points[7][0])
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[8][1])
        features.append(corner_points[8][0])
        features.append(corner_points[4][1])
        features.append(corner_points[4][0])
        features.append(corner_points[7][1] - corner_points[1][1])
        features.append(corner_points[7][0] - corner_points[1][0])
        features.append(corner_points[7][1] - corner_points[8][1])
        features.append(corner_points[7][0] - corner_points[8][0])
        features.append(corner_points[7][1] - corner_points[4][1])
        features.append(corner_points[7][0] - corner_points[4][0])
        features.append(corner_points[1][1] - corner_points[8][1])
        features.append(corner_points[1][0] - corner_points[8][0])
        features.append(corner_points[1][1] - corner_points[4][1])
        features.append(corner_points[1][0] - corner_points[4][0])
        features.append(corner_points[8][1] - corner_points[4][1])
        features.append(corner_points[8][0] - corner_points[4][0])
    elif database_path == '7/database/':
        print("this is test for 7")
        features.append(corner_points[3][1])
        features.append(corner_points[3][0])
        features.append(corner_points[11][1])
        features.append(corner_points[11][0])
        features.append(corner_points[15][1])
        features.append(corner_points[15][0])
        features.append(corner_points[16][1])
        features.append(corner_points[16][0])
        features.append(corner_points[3][1] - corner_points[11][1])
        features.append(corner_points[3][0] - corner_points[11][0])
        features.append(corner_points[3][1] - corner_points[15][1])
        features.append(corner_points[3][0] - corner_points[15][0])
        features.append(corner_points[3][1] - corner_points[16][1])
        features.append(corner_points[3][0] - corner_points[16][0])
        features.append(corner_points[11][1] - corner_points[15][1])
        features.append(corner_points[11][0] - corner_points[15][0])
        features.append(corner_points[11][1] - corner_points[16][1])
        features.append(corner_points[11][0] - corner_points[16][0])
        features.append(corner_points[15][1] - corner_points[16][1])
        features.append(corner_points[15][0] - corner_points[16][0])
    elif database_path == '6/database/':
        print("this is test for 6")
        features.append(corner_points[3][1])
        features.append(corner_points[3][0])
        features.append(corner_points[4][1])
        features.append(corner_points[4][0])
        features.append(corner_points[10][1])
        features.append(corner_points[10][0])
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[3][1] - corner_points[4][1])
        features.append(corner_points[3][0] - corner_points[4][0])
        features.append(corner_points[3][1] - corner_points[10][1])
        features.append(corner_points[3][0] - corner_points[10][0])
        features.append(corner_points[3][1] - corner_points[1][1])
        features.append(corner_points[3][0] - corner_points[1][0])
        features.append(corner_points[4][1] - corner_points[10][1])
        features.append(corner_points[4][0] - corner_points[10][0])
        features.append(corner_points[4][1] - corner_points[1][1])
        features.append(corner_points[4][0] - corner_points[1][0])
        features.append(corner_points[10][1] - corner_points[1][1])
        features.append(corner_points[10][0] - corner_points[1][0])
    elif database_path == '5/database/':
        print("this is test for 5")
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[2][1])
        features.append(corner_points[2][0])
        features.append(corner_points[3][1])
        features.append(corner_points[3][0])
        # features.append(corner_points[4][1])
        # features.append(corner_points[4][0])
        features.append(corner_points[1][1] - corner_points[2][1])
        features.append(corner_points[1][0] - corner_points[2][0])
        features.append(corner_points[1][1] - corner_points[3][1])
        features.append(corner_points[1][0] - corner_points[3][0])
        # features.append(corner_points[1][1] - corner_points[4][1])
        # features.append(corner_points[1][0] - corner_points[4][0])
        features.append(corner_points[2][1] - corner_points[3][1])
        features.append(corner_points[2][0] - corner_points[3][0])
        # features.append(corner_points[2][1] - corner_points[4][1])
        # features.append(corner_points[2][0] - corner_points[4][0])
        # features.append(corner_points[3][1] - corner_points[4][1])
        # features.append(corner_points[3][0] - corner_points[4][0])
    elif database_path == '4/database/':
        print("this is test for 4")
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[6][1])
        features.append(corner_points[6][0])
        features.append(corner_points[7][1])
        features.append(corner_points[7][0])
        # features.append(corner_points[12][1])
        # features.append(corner_points[12][0])
        features.append(corner_points[1][1] - corner_points[6][1])
        features.append(corner_points[1][0] - corner_points[6][0])
        features.append(corner_points[1][1] - corner_points[7][1])
        features.append(corner_points[1][0] - corner_points[7][0])
        # features.append(corner_points[1][1] - corner_points[12][1])
        # features.append(corner_points[1][0] - corner_points[12][0])
        features.append(corner_points[6][1] - corner_points[7][1])
        features.append(corner_points[6][0] - corner_points[7][0])
        # features.append(corner_points[6][1] - corner_points[10][1])
        # features.append(corner_points[6][0] - corner_points[10][0])
        # features.append(corner_points[7][1] - corner_points[12][1])
        # features.append(corner_points[7][0] - corner_points[12][0])
    elif database_path == '3/database/':
        print("this is test for 3")
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[2][1])
        features.append(corner_points[2][0])
        features.append(corner_points[6][1])
        features.append(corner_points[6][0])
        features.append(corner_points[9][1])
        features.append(corner_points[9][0])
        # features.append(corner_points[12][1])
        # features.append(corner_points[12][0])
        features.append(corner_points[1][0] - corner_points[2][0])  
        features.append(corner_points[1][1] - corner_points[2][1])
        features.append(corner_points[1][0] - corner_points[6][0])
        features.append(corner_points[1][1] - corner_points[6][1])
        features.append(corner_points[1][0] - corner_points[9][0])
        features.append(corner_points[1][1] - corner_points[9][1])
        # features.append(corner_points[1][0] - corner_points[12][0])
        # features.append(corner_points[1][1] - corner_points[12][1])
        features.append(corner_points[2][0] - corner_points[6][0])
        features.append(corner_points[2][1] - corner_points[6][1])
        features.append(corner_points[2][0] - corner_points[9][0])
        features.append(corner_points[2][1] - corner_points[9][1])
        # features.append(corner_points[2][0] - corner_points[12][0])
        # features.append(corner_points[2][1] - corner_points[12][1])
        features.append(corner_points[6][0] - corner_points[9][0])
        features.append(corner_points[6][1] - corner_points[9][1])
        # features.append(corner_points[9][0] - corner_points[12][0])
        # features.append(corner_points[9][1] - corner_points[12][1])
    elif database_path == '2/database/':
        print("this is test for 2")
        features.append(corner_points[2][1])
        features.append(corner_points[2][0])
        features.append(corner_points[6][1])
        features.append(corner_points[6][0])
        features.append(corner_points[8][1])
        features.append(corner_points[8][0])
        features.append(corner_points[12][1])
        features.append(corner_points[12][0])
        features.append(corner_points[14][1])
        features.append(corner_points[14][0])
        features.append(corner_points[17][1])
        features.append(corner_points[17][0])
        features.append(corner_points[2][1] - corner_points[6][1])
        features.append(corner_points[2][0] - corner_points[6][0])
        features.append(corner_points[2][1] - corner_points[8][1])
        features.append(corner_points[2][0] - corner_points[8][0])
        features.append(corner_points[2][1] - corner_points[12][1])
        features.append(corner_points[2][0] - corner_points[12][0])
        features.append(corner_points[2][1] - corner_points[14][1])
        features.append(corner_points[2][0] - corner_points[14][0])
        features.append(corner_points[2][1] - corner_points[17][1])
        features.append(corner_points[2][0] - corner_points[17][0])
        features.append(corner_points[6][1] - corner_points[8][1])
        features.append(corner_points[6][0] - corner_points[8][0])
        features.append(corner_points[6][1] - corner_points[12][1])
        features.append(corner_points[6][0] - corner_points[12][0])
        features.append(corner_points[6][1] - corner_points[14][1])
        features.append(corner_points[6][0] - corner_points[14][0])
        features.append(corner_points[6][1] - corner_points[17][1])
        features.append(corner_points[6][0] - corner_points[17][0])
        # features.append(corner_points[8][1] - corner_points[14][1])
        features.append(corner_points[8][1] - corner_points[12][1])
        features.append(corner_points[8][0] - corner_points[12][0])
        features.append(corner_points[8][1] - corner_points[14][1])
        features.append(corner_points[8][0] - corner_points[14][0])
        features.append(corner_points[8][1] - corner_points[17][1])
        features.append(corner_points[8][0] - corner_points[17][0])
        features.append(corner_points[12][1] - corner_points[14][1])
        features.append(corner_points[12][0] - corner_points[14][0])
        features.append(corner_points[12][1] - corner_points[17][1])
        features.append(corner_points[12][0] - corner_points[17][0])
        features.append(corner_points[14][1] - corner_points[17][1])
        features.append(corner_points[17][0] - corner_points[17][0])
    elif database_path == '1/database/':
        print("this is test for 1")
        features.append(corner_points[1][1])
        features.append(corner_points[1][0])
        features.append(corner_points[2][1])
        features.append(corner_points[2][0])
        features.append(corner_points[3][1])
        features.append(corner_points[3][0])
        features.append(corner_points[1][1] - corner_points[2][1])
        features.append(corner_points[1][0] - corner_points[2][0])
        features.append(corner_points[1][1] - corner_points[3][1])
        features.append(corner_points[1][0] - corner_points[3][0])
        features.append(corner_points[2][1] - corner_points[3][1])
        features.append(corner_points[2][0] - corner_points[3][0])
    return features


# 提取训练图像的特征
train_features = []
for file in train_database_files:
    features = extract_features(os.path.join(database_path, file))
    train_features.append(features)

for file in train_testcase_files:
    features = extract_features(os.path.join(testcase_path, file))
    train_features.append(features)

train_features = np.array(train_features)

# 計算每個特徵的平均值
mean_features = np.mean(train_features, axis=0)
std_features = np.std(train_features, axis=0)
std_features[std_features == 0] = 1
print(f"Mean Features: {mean_features}")
print(f"Std Features: {std_features}")
# 對每個特徵進行標準化
normalized_features = (train_features - mean_features)/std_features
# print(f"Normalized Features: {normalized_features}")
# # 打印所有训练图像的特征
# for idx, features in enumerate(normalized_features):
#     print(f"Image {idx+1} - Normalized Features: {features}")

# 提取測試圖像的特徵
test_features = []
for file in test_database_files:
    features = extract_features(os.path.join(database_path, file))
    test_features.append(features)

for file in test_testcase_files:
    features = extract_features(os.path.join(testcase_path, file))
    test_features.append(features)

test_features = np.array(test_features)

# 對測試數據進行標準化，使用訓練數據的平均值和標準差
normalized_test_features = (test_features - mean_features) / std_features

# # 打印所有測試圖像的特徵
# for idx, features in enumerate(normalized_test_features):
#     print(f"Image {idx+1} - Normalized Test Features: {features}")

# 创建训练和测试标签
train_labels = [1] * 25 + [2] * 25  # 假设前25个是类1，后25个是类2
test_labels = [1] * 25 + [2] * 25  # 假设前25个是类1，后25个是类2

# 训练 SVM 模型
model = svm.SVC(kernel='linear')
model.fit(normalized_features, train_labels)


# 打印训练过程中的支持向量数量
print(f"Number of support vectors: {len(model.support_)}")

# 进行预测
predictions = model.predict(normalized_test_features)



# # 打印每个样本的预测结果和实际标签
# for i, (pred, actual) in enumerate(zip(predictions, test_labels)):
#     print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, {'Correct' if pred == actual else 'Incorrect'}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(test_labels, predictions)
#show the matrix
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Real', 'Forged'], yticklabels=['Real', 'Forged'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()

TP = conf_matrix[0, 0]
TN = conf_matrix[1, 1]
FP = conf_matrix[1, 0]
FN = conf_matrix[0, 1]

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# 计算准确率
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
