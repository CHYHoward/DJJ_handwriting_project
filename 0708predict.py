# 5等分
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

# 定义文件路径
database_path = '9/database/'
testcase_path = '9/testcase/'
results_dir = '9'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
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
    
    features = np.array(features)
    non_zero_features = features[features != 0]
    mean_val = np.mean(non_zero_features)
    std_val = np.std(non_zero_features)
    
    normalized_features = np.zeros_like(features, dtype=float)
    for i in range(len(features)):
        if features[i] != 0:
            normalized_features[i] = (features[i] - mean_val) / std_val
    print(f"Extracted features: {normalized_features[:10]}")  # 
    return normalized_features

# 构建训练集和测试集

train_features = []
train_labels = []
test_features = []
test_labels = []

# 处理数据库图像
for file in train_database_files:
    path = os.path.join(database_path, file)
    features = extract_features(path)
    train_features.append(features)
    train_labels.append(1)

for file in test_database_files:
    path = os.path.join(database_path, file)
    features = extract_features(path)
    test_features.append(features)
    test_labels.append(1)

# 处理测试用例图像
for file in train_testcase_files:
    path = os.path.join(testcase_path, file)
    features = extract_features(path)
    train_features.append(features)
    train_labels.append(2)

for file in test_testcase_files:
    path = os.path.join(testcase_path, file)
    features = extract_features(path)
    print(f"Extracted features: {features[:10]}")  # 打印前5個特徵
    test_features.append(features)
    test_labels.append(2)

# 转换为 numpy 数组
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 训练 SVM 模型
model = svm.SVC(kernel='linear')
model.fit(train_features, train_labels)

# 打印训练过程中的支持向量数量
print(f"Number of support vectors: {len(model.support_)}")

# 进行预测
predictions = model.predict(test_features)


# 打印每个样本的预测结果和实际标签
for i, (pred, actual) in enumerate(zip(predictions, test_labels)):
    print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, {'Correct' if pred == actual else 'Incorrect'}")
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
# 计算混淆矩阵
conf_matrix = confusion_matrix(test_labels, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Forged'], yticklabels=['Real', 'Forged'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix-5-exclude.png'))
plt.show()
# 计算并打印 TP, TN, FP, FN
TP = conf_matrix[0, 0]
TN = conf_matrix[1, 1]
FP = conf_matrix[1, 0]
FN = conf_matrix[0, 1]

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")


accuracy_file_path = os.path.join(results_dir, 'accuracy-5-exclude.txt')
with open(accuracy_file_path, 'w') as file:
    file.write(f"Accuracy: {accuracy*100:.2f}%\n")



# # 10等分
# import os
# import cv2
# import numpy as np
# from sklearn import svm
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.utils import shuffle
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 定义文件路径
# database_path = '1/database/'
# testcase_path = '1/testcase/'
# results_dir = '1'
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
# # 读取图像文件列表，并确保仅选择 .bmp 文件
# database_files = sorted([f for f in os.listdir(database_path) if f.endswith('.bmp')])
# testcase_files = sorted([f for f in os.listdir(testcase_path) if f.endswith('.bmp')])

# # 随机打乱文件列表，确保数据分布随机
# database_files = shuffle(database_files, random_state=30)
# testcase_files = shuffle(testcase_files, random_state=30)

# # 将文件列表分割为训练集和测试集，各占 50%
# train_database_files = database_files[:25]
# test_database_files = database_files[25:]
# train_testcase_files = testcase_files[:25]
# test_testcase_files = testcase_files[25:]

# # 提取图像特征的函数
# def extract_features(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 将图像从 BGR 转换为 RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 计算图像的灰度值
#     intensity = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
#     # 根据灰度值判断是否为笔画（stroke）或背景
#     stroke_or_background = (intensity < 220).astype(int)
    
#     # 获取图像的高度和宽度
#     height, width = stroke_or_background.shape
#     # 水平和垂直方向的分割
#     horizontal_split = width // 10
#     vertical_split = height // 10
    
#     # 初始化特征列表
#     features = []
    
#     # 计算每个水平分区的笔画数量
#     for i in range(10):
#         start_col = i * horizontal_split
#         end_col = (i + 1) * horizontal_split if i < 9 else width
#         region = stroke_or_background[:, start_col:end_col]
#         stroke_count = np.sum(region)
#         features.append(stroke_count)
    
#     # 计算每个垂直分区的笔画数量
#     for i in range(10):
#         start_row = i * vertical_split
#         end_row = (i + 1) * vertical_split if i < 9 else height
#         region = stroke_or_background[start_row:end_row, :]
#         stroke_count = np.sum(region)
#         features.append(stroke_count)
    
#     # 转换为 numpy 数组
#     features = np.array(features)
#     # 过滤掉特征值为0的分区
#     non_zero_features = features[features != 0]
#     # 计算均值和标准差
#     mean_val = np.mean(non_zero_features)
#     std_val = np.std(non_zero_features)
    
#     # 归一化特征
#     normalized_features = np.zeros_like(features, dtype=float)
#     for i in range(len(features)):
#         if features[i] != 0:
#             normalized_features[i] = (features[i] - mean_val) / std_val
    
#     return normalized_features

# # 构建训练集和测试集
# train_features = []
# train_labels = []
# test_features = []
# test_labels = []

# # 处理数据库图像
# for file in train_database_files:
#     path = os.path.join(database_path, file)
#     features = extract_features(path)
#     train_features.append(features)
#     train_labels.append(1)

# for file in test_database_files:
#     path = os.path.join(database_path, file)
#     features = extract_features(path)
#     test_features.append(features)
#     test_labels.append(1)

# # 处理测试用例图像
# for file in train_testcase_files:
#     path = os.path.join(testcase_path, file)
#     features = extract_features(path)
#     train_features.append(features)
#     train_labels.append(2)

# for file in test_testcase_files:
#     path = os.path.join(testcase_path, file)
#     features = extract_features(path)
#     test_features.append(features)
#     test_labels.append(2)

# # 转换为 numpy 数组
# train_features = np.array(train_features)
# train_labels = np.array(train_labels)
# test_features = np.array(test_features)
# test_labels = np.array(test_labels)

# # 训练 SVM 模型
# model = svm.SVC()
# model.fit(train_features, train_labels)

# # 打印训练过程中的支持向量数量
# print(f"Number of support vectors: {len(model.support_)}")

# # 进行预测
# predictions = model.predict(test_features)

# # 计算准确率
# accuracy = accuracy_score(test_labels, predictions)
# print(f"Accuracy: {accuracy}")

# # 打印每个样本的预测结果和实际标签
# for i, (pred, actual) in enumerate(zip(predictions, test_labels)):
#     print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, {'Correct' if pred == actual else 'Incorrect'}")

# # 计算混淆矩阵
# conf_matrix = confusion_matrix(test_labels, predictions)

# # 可视化混淆矩阵
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Forged'], yticklabels=['Real', 'Forged'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# # plt.show()
# plt.savefig(os.path.join(results_dir, 'confusion_matrix-10-exclude.png'))
# # 计算并打印 TP, TN, FP, FN
# TP = conf_matrix[0, 0]
# TN = conf_matrix[1, 1]
# FP = conf_matrix[1, 0]
# FN = conf_matrix[0, 1]

# print(f"True Positives (TP): {TP}")
# print(f"True Negatives (TN): {TN}")
# print(f"False Positives (FP): {FP}")
# print(f"False Negatives (FN): {FN}")

# accuracy_file_path = os.path.join(results_dir, 'accuracy-10-exclude.txt')
# with open(accuracy_file_path, 'w') as file:
#     file.write(f"Accuracy: {accuracy*100:.2f}%\n")

# # print(f"Total features extracted: {total_features_count}")÷