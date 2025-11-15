# DJJ_handwriting_project
Handwriting Forgery Detection using DSP & SVM
Author: B10901163 張顥譽
📌 Overview
This project focuses on Chinese handwriting forgery detection.
Given an input handwritten character, the system extracts multiple DSP-based features and uses an SVM classifier to determine whether the handwriting is genuine or forged.
Handwriting authentication plays an important role in forensic analysis (e.g., credit card slips, wills, signatures). However, the natural variance within a single person’s handwriting makes the task challenging.
This project proposes a set of numerical features that capture stroke density, orientation, intensity, shape moments, and structural stability after erosion.

🎯 Goal
Use DSP-derived features + SVM to classify handwriting as genuine or forged.
Dataset
50 genuine samples
50 forged samples
Split evenly into:
Training: 25 genuine + 25 forged
Testing: 25 genuine + 25 forged

🖼️ Preprocessing
1. Read Image & Binarization
Convert the image to grayscale using:
Y = 0.299R + 0.587G + 0.114B
Rule to determine if a pixel belongs to a stroke:
If Y < 220 → Stroke
Else → Background
This produces a clean binary image ready for feature extraction.

📐 Feature Extraction
1. Projection Feature (10 features)
Split the image into 5 horizontal and 5 vertical segments.
Count stroke pixels in each segment.
Normalize mean & standard deviation → 10 features (p1–p10)
Accuracy: 90.22%

2. Moment Feature (9 features)
Based on classical image moments:
Let
B(i,j) = binarized pixel (1 = stroke, 0 = background)
m₀ and n₀ = centroid
mₐ,ᵦ = central moments
Extract the following 9 features:
m0, n0,
m20, m02, m11,
m30, m21, m12, m03
Accuracy: 89.55%

3. Intensity Feature (2 features)
From the grayscale Y:
Mean intensity of stroke pixels
Standard deviation of intensity
Accuracy: 88.89%

4. Stroke Stability After Erosion (3 features)
Binary erosion iteration:
Y0 = B
Yk(i,j) = Yk-1(i,j) AND neighbors (up, down, left, right)
Compute ratios for k = 1, 2, 3:
rk = (# stroke pixels after k erosions) / (# original stroke pixels)
Features: r1, r2, r3
Accuracy: 87.33%

5. Orientation Feature (3 features)
Compute centroid (x₀, y₀)
Build matrix of stroke coordinates → covariance → eigen decomposition
Extract:
θ = angle of horizontal eigenvector
λ_horizontal
λ_vertical
Accuracy: 76%

🧪 Experiments
Characters in the dataset include:
丁、建、均、五、十、伍、拾、務、實
Classification performance by character type:
Characters with < 8 strokes (丁、均、五、十、伍)
Using Projection + Moment + Intensity + Erosion features:
Character	Accuracy
丁	94%
均	96%
五	92%
十	94%
伍	96%
Characters with ≥ 8 strokes (建、拾、務、實)
Using Projection + Moment + Intensity + Erosion + Orientation:
Character	Accuracy
建	90%
拾	92%
務	94%
實	94%

✅ Conclusion
This project presents a DSP-based algorithm for Chinese handwriting forgery detection.
Due to the structural complexity of Chinese characters, multiple complementary features are required to accurately describe stroke distribution, intensity, shape, and orientation.
By selecting appropriate feature combinations depending on character complexity, the SVM classifier achieves an overall accuracy of:
🔥 93.57% Total Accuracy
This outperforms several existing handwriting verification methods.
