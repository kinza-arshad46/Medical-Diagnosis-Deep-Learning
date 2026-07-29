# 🩺 Medical Diagnosis Using Deep Learning

A deep learning-based web application for automated chest disease detection from Chest X-Ray images. This project uses Transfer Learning with ResNet50 to classify chest X-ray images and provides Grad-CAM visualizations for model interpretability. The application is built with Streamlit for an interactive and user-friendly interface.

---

## 📌 Project Overview

Medical image analysis plays a vital role in assisting healthcare professionals with early disease detection. This project aims to automate chest X-ray image classification using a deep learning model trained through transfer learning.

The system allows users to upload a Chest X-ray image, predicts the disease class, displays prediction confidence, and generates a Grad-CAM heatmap to highlight the regions responsible for the prediction.

---

## ✨ Features

- Chest X-ray image classification
- Transfer Learning using ResNet50
- Data preprocessing and augmentation
- Deep Learning-based prediction
- Grad-CAM visualization for explainable AI
- Prediction confidence score
- Confusion Matrix evaluation
- ROC Curve analysis
- Classification Report
- Interactive Streamlit Web Application
- Clean and modular project structure

---

## 🛠️ Technologies Used

### Programming Language

- Python 3.10

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pandas
- OpenCV

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Web Framework

- Streamlit

### Model Explainability

- Grad-CAM

### Utilities

- Scikit-learn
- Pillow
- Joblib
- YAML

---

## 📂 Project Structure

```
Medical-Diagnosis-Deep-Learning/
│
├── config/
│
├── logs/
│
├── outputs/
│
├── saved_models/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   ├── visualization/
│   └── training/
│
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
└── README.md
```

---

## 🧠 Model

The project uses **ResNet50**, a pre-trained Convolutional Neural Network (CNN), through Transfer Learning.

### Why ResNet50?

- Pre-trained on ImageNet
- Faster convergence
- High classification accuracy
- Reduces overfitting
- Suitable for medical image analysis

---

## 📊 Model Evaluation

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- Classification Report

---

## 🔥 Grad-CAM

To improve model interpretability, Grad-CAM is used to visualize the regions of the Chest X-ray that contributed most to the prediction.

This helps users understand the model's decision-making process.

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/kinza-arshad46/Medical-Diagnosis-Deep-Learning.git
```

Move into the project directory

```bash
cd Medical-Diagnosis-Deep-Learning
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💻 How to Use

1. Launch the Streamlit application.
2. Upload a Chest X-ray image.
3. The model preprocesses the image.
4. The trained ResNet50 model predicts the disease.
5. View:
   - Predicted Class
   - Confidence Score
   - Grad-CAM Heatmap
   - Evaluation Results

---

## 📈 Future Improvements

- Add DenseNet121 and EfficientNet models
- Multi-class disease classification
- PDF report generation
- Patient history integration
- Model comparison dashboard
- Cloud deployment enhancements
- Performance optimization

---


## 🤝 Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

## 📄 License

This project is developed for educational and research purposes.

---

## 👩‍💻 Author

**Kinza Arshad**

Data Science Student

GitHub:
https://github.com/kinza-arshad46

---

## ⭐ Support

If you found this project useful, don't forget to ⭐ Star the repository.
