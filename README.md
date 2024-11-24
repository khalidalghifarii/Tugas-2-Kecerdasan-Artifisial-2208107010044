# Tugas-2-Kecerdasan-Artifisial-[Muhammad Khalid Al Ghifari (2208107010044)]

## Informasi Proyek

- Jenis Kasus: Klasifikasi Gambar
- Dataset: Fashion MNIST
- Link Dataset: https://github.com/zalandoresearch/fashion-mnist

## Informasi Model

- Jenis Neural Network: Convolutional Neural Network (CNN)
- Jumlah Fitur: 784 (28x28 piksel)
- Jumlah Label: 10 kelas
- Optimisasi: Adam (learning rate: 0.001)
- Fungsi Aktivasi:
  - ReLU (Hidden Layers)
  - Softmax (Output Layer)
- Jumlah Hidden Layer: 3 Convolutional Blocks + 1 Dense Layer
- Jumlah Node per Layer:
  - Conv Block 1: 32 filters
  - Conv Block 2: 64 filters
  - Conv Block 3: 128 filters
  - Dense Layer: 256 nodes
- Total Parameters:
  Total params: 1,314,368 (5.01 MB)
  Trainable params: 437,738 (1.67 MB)
  Non-trainable params: 1,152 (4.50 KB)
  Optimizer params: 875,478 (3.34 MB)
 
## Struktur Repositori

- `fashion_mnist_cnn.ipynb`: Notebook utama
- `model_architecture.png`: Visualisasi arsitektur model
- `training_accuracy.png`: Plot akurasi training
- `training_loss.png`: Plot loss training
- `confusion_matrix.png`: Confusion matrix hasil evaluasi
- `fashion_mnist_model.h5`: Model yang telah dilatih
- `logs/`: Direktori log Tensorboard
- `model_summary.txt`: Ringkasan detail model

## Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn pandas
   ```
2. Buka dan jalankan notebook fashion_mnist_cnn.ipynb
3. Untuk melihat Tensorboard:
   ```bash
   tensorboard --logdir logs/fit
   ```
