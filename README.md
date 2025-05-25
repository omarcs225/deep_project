# 🧠 Multi-Class Segmentation from a Single-Class Conditional 3D UNet in TensorFlow

## 📌 Overview

This repository contains a TensorFlow implementation of a **3D UNet model** with **dense blocks** and **class-conditional segmentation**, allowing a single model to perform segmentation for multiple classes by conditioning on class-specific input.

---

## 🚀 Features

- Custom 3D UNet architecture  
- Dense blocks for enhanced feature reuse  
- Conditional segmentation using external class indicators  
- Easily scalable to any number of classes  
- Designed for volumetric data (e.g., medical imaging)  

---

## 🧱 Architecture

- **DenseBlock3D**: Implements a dense connection block for 3D volumes.
- **UNet3D_Dense_Cond**: Full encoder-decoder structure with skip connections and conditional input support.

---

## 🖼️ Conditional Segmentation

The model leverages a conditioning value (`cond_values`) for each class. It appends a constant tensor (with the class-specific value) to the input at **each stage of the network**, guiding the segmentation process for the selected class.

---

## 🛠️ Installation

```bash
pip install tensorflow
from your_model_file import UNet3D_Dense_Cond
import tensorflow as tf

# Initialize model
model = UNet3D_Dense_Cond(in_channels=1, num_classes=4, cond_values=[0.2, 0.4, 0.6, 0.8])

# Input tensor shape: [batch, depth, height, width, channels]
input_tensor = tf.random.normal([1, 64, 128, 128, 1])

# Forward pass for class index 2
output = model(input_tensor, class_idx=2)

print("Output shape:", output.shape)
## 🧪 Testing

To run the test script and see the model output, use:

```bash
python MultiClass\ Segemention\ from\ Single\ Class\ TenserFlow.py
Output shape: (1, 64, 128, 128, 1)
---

## 🛠️ Results Comparison
| Model                           | Avg. Dice Score | Num. of Models | Parameters | Class Conditioning | Notes                        |
| ------------------------------- | --------------- | -------------- | ---------- | ------------------ | ---------------------------- |
| Chaudhari et al. (MICCAI 2019)  | 0.82            | 1              | \~15M      | Yes                | 2D slices, class token input |
| **Ours (3D Cond-UNet + Dense)** | 0.85            | 1              | \~18M      | Yes                | Full 3D volumes, cond tensor |
