# 🧬 Skin Cancer Classification using Deep Learning

This project focuses on building a **Skin Cancer Classification Model** using deep learning techniques. It utilizes the **HAM10000** dataset to classify different types of skin lesions based on dermatoscopic images. The goal is to assist in early and accurate detection of skin cancer using AI-driven solutions.

## 📌 Project Description

Skin cancer is one of the most common types of cancer worldwide. Early detection and proper diagnosis are crucial for effective treatment. This project leverages a convolutional neural network (CNN), specifically a pre-trained **DenseNet** architecture, to classify skin lesions into seven categories. The model is trained, validated, and evaluated on the HAM10000 dataset.

### 🔍 Skin Lesion Categories:
- Melanocytic nevi
- Melanoma
- Benign keratosis-like lesions
- Basal cell carcinoma
- Actinic keratoses
- Vascular lesions
- Dermatofibroma

## 📂 Dataset

The dataset used in this project is publicly available on Kaggle:

**[HAM10000 Dataset - Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**

It contains over 10,000 dermatoscopic images with metadata and diagnostic labels.

## 🚀 Technologies Used

- Python 🐍
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib / Seaborn
- Scikit-learn
- DenseNet (Transfer Learning)


## 📈 Model Performance

The model was evaluated using accuracy, precision, recall, and F1-score. It shows promising results on validation and test data.

## 💡 Future Improvements

- Deploy the model as a web app using Flask or Streamlit
- Integrate patient metadata for better accuracy
- Expand dataset for real-world generalization
