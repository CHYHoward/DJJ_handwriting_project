from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

image = cv2.imread('1/database/base_1_1_1.bmp')  # 以灰度模式讀取圖像

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 计算强度
intensity = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]

# 判定stroke或背景
stroke_or_background = (intensity < 220).astype(int)
B = stroke_or_background

# 获取图像的尺寸
rows, cols = B.shape
# 获取所有stroke像素的坐标
stroke_coords = np.column_stack(np.where(B == 1))

# 用于存储边界点坐标
boundary_points = []
# 创建一个与原图像相同大小的图像用于显示结果
output_image = image_rgb.copy()

# 遍历每个像素
for m in range(1, rows - 1):
    for n in range(1, cols - 1):
        if B[m, n] == 1 and (
            B[m-1, n] == 0 or B[m+1, n] == 0 or
            B[m, n-1] == 0 or B[m, n+1] == 0
        ):
            # 标记边界点为红色
            output_image[m, n] = [255, 0, 0]
            
# 遍历所有的stroke坐标
for (m, n) in stroke_coords:
    # 检查当前点的上下左右是否是背景
    if (m > 0 and B[m-1, n] == 0) or (m < B.shape[0]-1 and B[m+1, n] == 0) or \
       (n > 0 and B[m, n-1] == 0) or (n < B.shape[1]-1 and B[m, n+1] == 0):
        boundary_points.append((m, n))

# 打印边界点坐标和总个数
print("Boundary points:")
for point in boundary_points:
    print(point)
print(f"Total number of boundary points: {len(boundary_points)}")
# 显示结果
plt.imshow(output_image)
plt.axis('off')
plt.show()

np.savetxt('boundary_points.txt', boundary_points, fmt='%d')
#把這些點當成x,y座標，畫成圖
boundary_points = np.array(boundary_points)
x = boundary_points[:, 1]
y = boundary_points[:, 0]
plt.scatter(x, y)
plt.gca().invert_yaxis()
plt.show()
