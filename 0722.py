from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# 定义文件路径
database_path = '9/database/'
testcase_path = '9/testcase/'

# 读取图像文件列表
database_files = sorted([f for f in os.listdir(database_path) if f.endswith('.bmp')])
testcase_files = sorted([f for f in os.listdir(testcase_path) if f.endswith('.bmp')])

# 随机打乱文件列表
database_files = shuffle(database_files, random_state=50)
testcase_files = shuffle(testcase_files, random_state=50)

# 分割文件列表为训练集和测试集
train_database_files = database_files[:25]
test_database_files = database_files[25:]
train_testcase_files = testcase_files[:25]
test_testcase_files = testcase_files[25:]

def erode_image(binary_image):
    eroded_image = np.copy(binary_image)
    for i in range(1, binary_image.shape[0] - 1):
        for j in range(1, binary_image.shape[1] - 1):
            eroded_image[i, j] = binary_image[i, j] & binary_image[i, j+1] & binary_image[i, j-1] & binary_image[i+1, j] & binary_image[i-1, j]
    return eroded_image



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
# print(f"Mean Features: {mean_features}")
# print(f"Std Features: {std_features}")
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
print (database_path)