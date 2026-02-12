# MNIST Digit Classifier (PyTorch)

## 📌 Project Objective
This project trains a neural network to classify handwritten digits (0–9) 
using the MNIST dataset.

The model is trained locally using PyTorch.

---

## 🧠 Model Architecture
- Input: 28x28 grayscale image (784 features)
- Fully Connected Layers:
  - 784 → 128
  - 128 → 64
  - 64 → 10
- Activation Function: ReLU
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 5
- Batch Size: 64

---

## 🎯 Results
- Training Accuracy: ~98%
- Test Accuracy: ~97%

---

## ▶️ How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run training:
   python train.py

---

## 💾 Output
The model saves:
- mnist_model_complete.pth
- mnist_model_state_dict.pth
- mnist_model_checkpoint.pth
