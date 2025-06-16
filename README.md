# 🧠 Brain Tumor Detection using CNN

A deep learning-based solution that classifies MRI brain scans to detect the presence of a tumor. Built using Convolutional Neural Networks (CNN), this project aims to assist early and reliable detection of brain tumors through automated medical image analysis.

---

## 🔬 Project Summary

- Classifies MRI images into **Tumor** or **No Tumor**
- Trained on real MRI scan datasets
- Achieved finalist position in **Neural Nexus AI Hackathon (2025)**
- Uses image preprocessing, CNN model, and evaluation metrics for performance

---

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib
- **Tools:** Jupyter Notebook / Colab, scikit-learn
- **Dataset:** Brain MRI Images from Kaggle (or equivalent)

---

## 📁 Project Structure


---

## 🔄 How It Works

1. **Image Preprocessing**
   - Grayscale conversion
   - Resizing to fixed dimensions (e.g., 150x150)
   - Normalization

2. **Model Architecture**
   - Convolutional layers with ReLU
   - MaxPooling layers
   - Dense fully connected layers
   - Output layer with sigmoid (binary classification)

3. **Training & Evaluation**
   - Trained using binary cross-entropy loss
   - Evaluated on accuracy, precision, recall

---

## 📈 Results

- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~95%
- **Model Type:** Binary Classifier (Tumor / No Tumor)
- **Confusion Matrix, ROC Curve** plotted for interpretability

---

## 🔍 Sample Prediction

```python
img = load_img("sample_mri.jpg", target_size=(150, 150))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print("Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor")

