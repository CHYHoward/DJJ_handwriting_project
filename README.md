# Handwriting Forgery Detection using DSP & SVM  
Author: **B10901163 張顥譽**

## 📌 Overview
This project focuses on **Chinese handwriting forgery detection** using DSP-based feature extraction and an SVM classifier.
**(detail report and code can be seen in main branch)**

Handwriting authentication is widely used in forensic document analysis (e.g., credit card slips, bills, wills). However, handwriting varies naturally even within the same person, making forgery detection challenging.

This project introduces multiple numerical features that describe stroke density, intensity, geometry, and orientation to determine whether a handwritten character is **genuine** or **forged**.

---

## 🎯 Goal
Use DSP-derived features + SVM to classify handwriting as **genuine** or **forged**.

### Dataset
- 50 genuine samples  
- 50 forged samples  
- Splits:
  - **Training:** 25 genuine + 25 forged  
  - **Testing:** 25 genuine + 25 forged  

---

## 🖼️ Image Preprocessing
### 1. Binarization
Convert image to grayscale:
Y = 0.299R + 0.587G + 0.114B
A pixel is classified as a stroke pixel if: 
Y < 220 → stroke
Y ≥ 220 → background

This produces a clean binary image for analysis.

---

## 📐 Feature Extraction

### 1. Projection Feature (10 features)
- Divide the image into **5 horizontal** and **5 vertical** regions.
- Count stroke pixels in each region.
- Normalize mean & standard deviation to obtain **10 features**.

**Accuracy:** 90.22%

---

### 2. Moment Feature (9 features)
Using central moments:

- Let B(i,j) ∈ {0,1}  
- Compute centroid (m₀, n₀)  
- Extract:
m0, n0,
m20, m02, m11,
m30, m21, m12, m03

**Accuracy:** 89.55%

---

### 3. Intensity Feature (2 features)
Using grayscale Y:
- Mean intensity of stroke pixels  
- Standard deviation of stroke intensity  

**Accuracy:** 88.89%

---

### 4. Stroke Stability After Erosion (3 features)
Binary erosion:
Y0 = B

Yk(i,j) = Yk-1(i,j) AND neighbors (up, down, left, right)

Define ratio:
rk = (# stroke pixels after k erosions) / (# original stroke pixels)


Extract **r1, r2, r3**.

**Accuracy:** 87.33%

---

### 5. Orientation Feature (3 features)
- Compute stroke centroid (x₀, y₀)
- Build coordinate matrix and perform eigen decomposition  
- Extract:
  - θ (angle of dominant eigenvector)
  - λ_horizontal  
  - λ_vertical  

**Accuracy:** 76%

---

## 🧪 Experiments

Dataset characters:  
**丁、建、均、五、十、伍、拾、務、實**

### Characters with < 8 strokes  
Using **Projection + Moment + Intensity + Erosion features**:

| Character | Accuracy |
|----------|----------|
| 丁 | 94% |
| 均 | 96% |
| 五 | 92% |
| 十 | 94% |
| 伍 | 96% |

### Characters with ≥ 8 strokes  
Using **All features including Orientation**:

| Character | Accuracy |
|----------|----------|
| 建 | 90% |
| 拾 | 92% |
| 務 | 94% |
| 實 | 94% |

---

## ✅ Conclusion
This project presents a DSP-based algorithm for **Chinese handwriting forgery detection**.  
Because Chinese characters have complex structures, multiple features are required to capture full handwriting characteristics.

By adapting feature selection based on stroke count, the SVM classifier achieves:

# **🔥 Overall Accuracy: 93.57%**

This performance surpasses many existing forgery detection methods.

---

## 📂 Features Used Summary
- Projection (10)
- Moments (9)
- Intensity (2)
- Stroke Erosion Stability (3)
- Orientation (3)

Total: **27 features**

---



