# 🌱 Early Exit Deep Learning for Sustainable AI  
### 🚀 Confidence-Based Dynamic Inference to Reduce Computation

---

## 📌 Overview
This project implements **Early Exit Neural Networks**, a technique that enables **dynamic inference** by allowing models to stop computation early when confident enough.

Traditional deep learning models process all inputs fully, leading to unnecessary computation. This project introduces **confidence-based exits** to improve efficiency while maintaining accuracy.

📄 Based on research work:  
**"Early Exit Deep Learning: A Pathway to Sustainable AI through Reduced Computational Footprint"**

---

## 🧠 Key Idea: Early Exit Networks

![Early Exit Architecture](https://via.placeholder.com/800x400?text=Early+Exit+Architecture)

- Intermediate layers produce predictions  
- If confidence ≥ threshold → exit early  
- Otherwise → continue deeper  

✅ Saves computation  
✅ Reduces energy usage  
✅ Speeds up inference  

---

## 🎯 Objectives
- ⚡ Reduce inference time  
- 💻 Minimize computational cost (FLOPs)  
- 📊 Maintain high accuracy  
- 🌍 Promote Green AI  

---

## 🏗️ Architectures Implemented

### 1️⃣ Custom CNN
![CNN](https://via.placeholder.com/400?text=CNN+Architecture)

- 3 convolutional blocks  
- Early exits after each block  
- Best for studying efficiency trade-offs  

---

### 2️⃣ ResNet-18 (Transfer Learning)
![ResNet](https://via.placeholder.com/400?text=ResNet+Architecture)

- Pretrained on ImageNet  
- Partial fine-tuning  
- Demonstrates limitations of frozen layers  

---

### 3️⃣ LSTM (Sequential Model)
![LSTM](https://via.placeholder.com/400?text=LSTM+Architecture)

- Treats image as sequence  
- Shows architectural mismatch  
- Low early-exit effectiveness  

---

### 4️⃣ Vision Transformer (ViT-B/16) ⭐
![ViT](https://via.placeholder.com/400?text=Vision+Transformer)

- Pretrained transformer model  
- Early exits at transformer layers  
- Best performance achieved  

---

## ⚙️ Methodology

### 📌 Early Exit Condition
\[
\max_c p_k(c|x) \geq \tau
\]

- \( \tau \) = confidence threshold (0.50–0.99)  
- Exit when model is confident  

---

### 📉 Training Loss
\[
L = L_{final} + 0.5L_1 + 0.3L_2 + 0.2L_3
\]

- Encourages early predictions  
- Maintains final accuracy  

---

## 📊 Results

### 🔥 Accuracy Comparison
| Model        | Accuracy |
|-------------|---------|
| CNN         | 71.2%   |
| ResNet-18   | 92.5%   |
| LSTM        | 60.3%   |
| ViT-B/16 ⭐ | **97.5%** |

---

### ⚡ FLOPs Reduction
| Model | Max Reduction |
|------|--------------|
| CNN  | **35%** |
| ViT  | 18.8% |
| ResNet | ~2% |
| LSTM | ~1% |

---

### 📈 Accuracy vs Efficiency Trade-off
![Tradeoff](https://via.placeholder.com/800x400?text=Accuracy+vs+FLOPs)

---

## 🔍 Key Insights

- ✅ Pretrained models enable effective early exits  
- ❌ Frozen layers reduce confidence calibration  
- ⚠️ Poor architectures → no benefit (LSTM case)  
- 📉 CNN shows best interpretable trade-off  

---

## 🌍 Green AI Impact

- Up to **35% computation savings**  
- Minimal accuracy loss  
- Suitable for:
  - Edge devices  
  - Real-time systems  
  - Energy-efficient AI  

---
## ▶️ How to Run

```bash
git clone https://github.com/your-username/early-exit-networks.git
cd early-exit-networks
pip install torch torchvision numpy matplotlib
jupyter notebook
