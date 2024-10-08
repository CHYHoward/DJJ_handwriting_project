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

# 用于统计特征数量的计数器
total_features_count = 0
non_zero_features_count = 0

# 提取特征的函数
def extract_features(image_path):
    global non_zero_features_count
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
    
    if len(non_zero_features) == 0:
        return features  # 如果所有特征都为零，返回原特征（不会影响 SVM 训练，因为特征全部为零）

    mean_val = np.mean(non_zero_features)
    std_val = np.std(non_zero_features)
    
    normalized_features = (features - mean_val) / std_val
    
    # # 计算矩特征
    # m_values, n_values = np.meshgrid(np.arange(width), np.arange(height))
    # m0 = np.sum(m_values * stroke_or_background) / np.sum(stroke_or_background)
    # n0 = np.sum(n_values * stroke_or_background) / np.sum(stroke_or_background)
    
    # # 定义矩特征计算函数
    # def moment_feature(a, b):
    #     return np.sum(((m_values - m0) ** a) * ((n_values - n0) ** b) * stroke_or_background) / np.sum(stroke_or_background)
    
    # # 计算并添加指定的矩特征
    # moment_features = [
    #     moment_feature(2, 0),  # m2,0
    #     moment_feature(0, 2),  # m0,2
    #     moment_feature(1, 1),  # m1,1
    #     moment_feature(3, 0),  # m3,0
    #     moment_feature(2, 1),  # m2,1
    #     moment_feature(1, 2),  # m1,2
    #     moment_feature(0, 3)   # m0,3
    # ]
    
    B = stroke_or_background
    M00 = np.sum(B)
    print(M00)
    M10 = np.sum(np.arange(width) * np.sum(B, axis=0))
    M01 = np.sum(np.arange(height) * np.sum(B, axis=1))

    m0 = M01 / M00
    n0 = M10 / M00
    # print(m0)
    # print(n0)
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
    # print(m03)

    # 添加矩特征到特征向量
    final_features = np.concatenate((normalized_features, [m0, n0, m02, m11, m20, m21, m30,m03, m12]))
    
    # 统计特征数量
    global total_features_count
    total_features_count += len(features)
    non_zero_features_count += len(non_zero_features)
    # print (len(final_features))
    return final_features

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
    test_features.append(features)
    test_labels.append(2)

# 转换为 numpy 数组
train_features = np.array(train_features)
train_labels = np.array(train_labels)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# 训练 SVM 模型
model = svm.SVC()
model.fit(train_features, train_labels)

# 打印训练过程中的支持向量数量
print(f"Number of support vectors: {len(model.support_)}")

# 进行预测
predictions = model.predict(test_features)

# 计算准确率
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")

# 打印每个样本的预测结果和实际标签
for i, (pred, actual) in enumerate(zip(predictions, test_labels)):
    print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, {'Correct' if pred == actual else 'Incorrect'}")

# 计算混淆矩阵
conf_matrix = confusion_matrix(test_labels, predictions)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Forged'], yticklabels=['Real', 'Forged'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, 'confusion_matrix-0715-1.png'))

# 计算并打印 TP, TN, FP, FN
TP = conf_matrix[0, 0]
TN = conf_matrix[1, 1]
FP = conf_matrix[1, 0]
FN = conf_matrix[0, 1]

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# 打印特征统计结果
print(f"Total features extracted: {total_features_count}")
print(f"Non-zero features: {non_zero_features_count}")

accuracy_file_path = os.path.join(results_dir, 'accuracy-0715-1.txt')
with open(accuracy_file_path, 'w') as file:
    file.write(f"Accuracy: {accuracy*100:.2f}%\n")
