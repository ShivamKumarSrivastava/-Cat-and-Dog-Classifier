# 🐶🐱 Cat vs Dog Image Classification

A deep learning project that classifies images as **Cat** or **Dog** using **TensorFlow**. The dataset is sourced from **Kaggle**, and the entire workflow is built, trained, and deployed using **Lightning AI**. The repository is connected with **GitHub** for continuous updates and improvements.

---

## 🚀 Project Overview

This project trains a convolutional neural network (CNN) to distinguish between cats and dogs. It includes:

* Data preprocessing
* Model building (TensorFlow / Keras)
* Training & validation
* Prediction on custom images
* Code development on Lightning AI
* Version control using GitHub

---

## 📦 Dataset

* Source: **Kaggle Cats and Dogs Dataset**
* Contains thousands of labeled cat and dog images
* Dataset is organized for training and validation splits

> Note: Download instructions for the dataset may vary; ensure you have access via Kaggle.

---

## 🧠 Model Architecture

This model is built using a custom CNN with layers such as:

* Convolutional Layers
* Max Pooling Layers
* Batch Normalization
* Dense Layers
* Dropout for regularization

Training is done using `binary_crossentropy` loss and the **Adam** optimizer.

---

## 🛠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Lightning AI** (development environment)
* **GitHub** (version control)
* **NumPy**, **Matplotlib**, **ImageDataGenerator**

---

## 🔧 How to Run the Project

### 1. Clone the repository

```bash
gh repo clone ShivamKumarSrivastava/-Cat-and-Dog-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```python
python cli.py
```

### 4. Test on a new image

```python
python cli.py --img path/to/image.jpg
```

---

## 📊 Model Performance

* Training Accuracy: ~ >0.82
* Validation Accuracy: ~ >0.6
---

## 📁 Project Structure

```
├── data/
├── models/
│   └── dog_cat_final_model.keras
├── cli.py
├── streamlit_app.py
├── README.md
└── requirements.txt
```

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork the repo, create a branch, and submit a pull request.

---

## ⭐ Acknowledgements

* Kaggle for providing the dataset
* TensorFlow team
* Lightning AI for coding environment

---

If you like this project, don't forget to **star ⭐ the repository!**
