# Breast Cancer Classification using Neural Networks

This project is a deep learning–based system to classify breast tumors as **malignant** or **benign** using a neural network.  
It is built using **TensorFlow, Keras, and Scikit-learn** and trained on the **Wisconsin Breast Cancer Dataset**.

The goal of this project is to understand how neural networks can be applied to real medical data and to build a complete end-to-end machine learning pipeline.

---

## 📌 Dataset

The model uses the **Wisconsin Breast Cancer Dataset**, which is available directly through `sklearn.datasets`.

It contains:
- **569 samples**
- **30 numeric features** extracted from digitized images of breast mass
- Target labels:
  - `0` → Malignant  
  - `1` → Benign  

---

## 🧠 Model Architecture

The neural network used in this project:

- Input layer with 30 features  
- Two hidden layers:
  - Dense (32 neurons, ReLU)
  - Dense (16 neurons, ReLU)
- Output layer:
  - Dense (1 neuron, Sigmoid) for binary classification  

The model is trained using:
- **Adam optimizer**
- **Binary Cross-Entropy loss**

---

## 📊 Model Performance

After training for 30 epochs, the model achieves:

- **Test Accuracy ≈ 96–97%**

This shows the neural network is able to learn meaningful patterns from the medical data.

---

## 🔍 How Prediction Works

The model outputs a probability:

- Values close to `0` → Malignant  
- Values close to `1` → Benign  

A threshold of `0.5` is used to determine the final class.

---

## 🛠 How to Run This Project

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd "Breast Cancer Classification with NN (DL)"

## CREATE VIRTUAL ENVIRONMENT
python -m venv venv
source venv/bin/activate


---


## Install dependencies
pip install -r requirements.txt

---

## Run the model
python model/deep_learning_breast_cancer_classification_with_nn.py

The script will:
	•	Train the neural network
	•	Evaluate test accuracy
	•	Run a prediction on a sample input

---

##  Project Structure
Breast Cancer Classification with NN (DL)
│
├── model/
│   ├── deep_learning_breast_cancer_classification_with_nn.py
│   └── testingTensorFlow.py
│
├── data/
│   └── data.csv   (optional, not used in current training)
│
├── venv/
├── requirements.txt
└── README.md

---

### Notes
	•	The dataset is loaded directly from sklearn.datasets, so no manual data download is required.
	•	A virtual environment is used to keep all dependencies isolated and reproducible.

⸻

## Future Improvements

Some possible future upgrades:
	•	Save and load the trained model instead of retraining every time
	•	Add confusion matrix and ROC curve
	•	Build a web interface for predictions

---

## Author

Vansh

This is my first deep learning project, built to understand how neural networks work on real-world medical data.
---

If you want, next I can also help you write a **GitHub description**, or a **LinkedIn caption** when you publish this.