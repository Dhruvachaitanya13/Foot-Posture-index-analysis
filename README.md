# Foot-Posture-index-analysis
This project leverages Convolutional Neural Networks (CNNs) to detect and classify foot postures from grayscale foot images. It calculates the **Foot Posture Index (FPI)** using geometric landmarks and automates foot type classification: **Pronated**, **Neutral**, or **Supinated**.

## 📁 Project Structure

```
Final-Project-AI-Foot-Posture-Index/
│
├── dataset/                      # Contains train/test images
├── model/                        # Trained CNN model will be saved here
├── results/                      # Results of predictions
├── Foot_Posture_Index_CNN.ipynb  # Main notebook
├── helper_functions.py           # Image processing utilities
├── predict.py                    # CLI script for inference
├── requirements.txt              # List of dependencies
└── README.md
```

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ozgurcetinpy/Final-Project-AI-Foot-Posture-Index.git
cd Final-Project-AI-Foot-Posture-Index
```

### 2. Set Up Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If the `requirements.txt` is missing, use the dependencies below:

```bash
pip install numpy opencv-python tensorflow keras matplotlib seaborn scikit-learn
```

## 🧠 Model Overview

- **Model Type**: CNN (Keras / TensorFlow)
- **Input**: Grayscale foot images
- **Output**: Class label – *Pronated*, *Neutral*, or *Supinated*
- **Landmark Detection**: Extracts key points (e.g., heel, forefoot) to calculate geometric angles and arch index.

## 🚀 Usage

### ▶️ Option 1: Run the Notebook

Open the notebook in Jupyter or Colab:

```bash
jupyter notebook Foot_Posture_Index_CNN.ipynb
```

Follow the steps inside to train, evaluate, and test the model.

### ▶️ Option 2: Predict from Command Line

If you have trained the model and saved it (or downloaded it to the `model/` directory), you can run predictions on new images:

```bash
python predict.py --image path/to/image.jpg
```

This will:
- Preprocess the input image
- Load the trained CNN model
- Output the predicted foot posture

## 📊 Model Training

To retrain the model:
1. Place your training images in `dataset/train/` and test images in `dataset/test/`
2. Run all cells in the notebook `Foot_Posture_Index_CNN.ipynb`
3. The trained model will be saved under `model/` directory.

## 📁 Data Format

Each image should:
- Be in grayscale (or convert to grayscale)
- Have clear visibility of the foot (sole view preferred)
- Be labeled properly if used for training

Expected directory format:

```
dataset/
├── train/
│   ├── pronated/
│   ├── neutral/
│   └── supinated/
├── test/
│   ├── pronated/
│   ├── neutral/
│   └── supinated/
```

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- Visualization of landmark detection and angle measurement

## 🔗 Related Projects

- [FPI Research Paper (External)](https://pubmed.ncbi.nlm.nih.gov/20052517/)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)
