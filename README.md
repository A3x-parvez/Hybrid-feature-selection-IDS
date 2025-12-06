# 💠 Hybrid-feature-selection-IDS
### *Deep Learning–Based Intrusion Detection System with PCA / ICA / FA Hybrid Feature Fusion*

This project implements a complete Intrusion Detection System (IDS) pipeline that combines:

- **Hybrid feature selection** using **PCA**, **ICA**, and **Factor Analysis (FA)**
- **Statistical scoring** using Mutual Information, Chi-Square, ANOVA F-test
- **Deep Learning models**: LSTM, GRU, CNN, ANN
- **Automatic evaluation across multiple feature subset sizes**
- **GPU-optimized training with memory growth**
- **Automatic Word report generation per fusion method**

A single, unified Python script executes the full pipeline from preprocessing → feature fusion → model training → evaluation → report generation.

---

# 📌 Key Features

### ✅ Hybrid Feature Selection  
The system evaluates feature importance using MI, Chi2, and F-score, then fuses these using:
- **PCA (Principal Component Analysis)**
- **ICA (Independent Component Analysis)**
- **FA (Factor Analysis)**

Each fusion method produces a ranked list of features based on a computed meta-score.

---

### ✅ Deep Learning Model Suite

| Model | Purpose |
|-------|---------|
| **LSTM** | Captures long-range flow dependencies |
| **GRU** | Light-weight alternative to LSTM |
| **CNN 1D** | Extracts spatial patterns |
| **ANN** | Dense network baseline |

Each model is trained & tested using multiple feature subset sizes (`k = 5 → 60`).

---

### ✅ Automatic Word Report Generation
For each feature fusion method, the system creates a `.docx` file with:

- A table of all evaluated `k` values  
- Train/Test accuracy  
- Train/Test F1-score  
- Selected features  
- Best performing `k` for each model  

Generated output files:

```
ALL_MODEL_RESULTS_PCA.docx
ALL_MODEL_RESULTS_ICA.docx
ALL_MODEL_RESULTS_FA.docx
```

---

### ✅ GPU-Optimized + Dimension Safe
The script:

- Auto-detects GPU  
- Enables memory growth  
- Handles datasets with any number of features  
- Automatically adjusts invalid `k` values  
- Auto-detects binary target column  

---

# 📂 Project Structure

```
Hybrid-feature-selection-IDS/
│
├── README.md                      # Documentation
├── requirements.txt               # Requirment files for the env
├── CIC_IDS_Training.py            # Main pipeline script
├── CIC_IDS_Training_gen.py        # Reformat pipeline script AI (NOT Tested)
├── CIC-IDS-2017_fin_capped.csv    # Dataset
└── model_results/
    ├── ALL_MODEL_RESULTS_PCA.docx
    ├── ALL_MODEL_RESULTS_ICA.docx
    └── ALL_MODEL_RESULTS_FA.docx
```

---

# 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/A3x-parvez/Hybrid-feature-selection-IDS.git
cd Hybrid-feature-selection-IDS
```

## 🔧 **Environment Setup & Dependency Installation**

This project was developed using:

```
Python 3.12.3
```

It is recommended to create an isolated virtual environment before installation.

---

### **1️⃣ Create Virtual Environment**

#### **Windows**

```bash
python -m venv Hybrid_env
```

#### **Linux / macOS**

```bash
python3 -m venv Hybrid_env
```

---

### **2️⃣ Activate the Environment**

#### **Windows**

```bash
Hybrid_env\Scripts\activate
```

#### **Linux / macOS**

```bash
source Hybrid_env/bin/activate
```

---

### **3️⃣ Install Dependencies**

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

### **⚡ (Optional) GPU Support**

If your system supports CUDA and you want GPU acceleration:

```bash
pip install tensorflow-gpu
```

Verify GPU availability:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### **GLOBAL CONFIGURATION**

The following configuration values are used as sensible defaults. Adjust them for quick tests or full production runs.

```python
# Which fusion methods to run
FUSION_LIST = ['pca', 'ica', 'fa']

# Models to evaluate (order doesn't matter)
MODEL_LIST = ['LSTM', 'GRU', 'CNN', 'ANN']

# Data split
TEST_SIZE = 0.2            # Fraction of data reserved for testing
RANDOM_STATE = 42

# Training settings
TRAINING_EPOCHS = 20        # Use a small value for quick tests; increase (e.g. 50+) for final runs
BATCH_SIZE = 512

# Feature-selection k upper limit
# Use a small value for development/testing; set to 65 for full experiments
RANGE_LIMIT = 65
```

---


# ▶️ How to Run

Run the system with your dataset:

```bash
python script.py dataset.csv
```

Example:

```bash
python CIC_IDS_Training.py CIC-IDS-2017_fin_capped.csv
```

---

# 📌 Dataset Requirements

Your dataset must include:

* **A binary target column** (`0 = benign`, `1 = attack`)
<!-- * The system auto-detects the column name (`target`, `label`, `class`, etc.) -->
* For this Script the dataset target column should be binary class , encoded fromat (0,1) and name should be "target" .
* All input features must be numeric

---

# 🧠 Pipeline Overview

### **1. Preprocessing**

* Remove NaN/inf values
* Extract features + target
* Data should be preprocessed if needed mode preprocessing.

### **2. Feature Scoring**

Computes:

* Mutual Information
* Chi-Square
* F-score

### **3. Hybrid Feature Fusion**

Depending on the selected method:

* PCA → meta-feature component
* ICA → independent component (sign-corrected)
* FA → latent factor score

### **4. Deep Learning Training**

For each model and each k:

* Select top-k features
* Split data (train/test)
* Train the model
* Track accuracy & F1

### **5. Report Generation**

All results stored in DOCX tables with performance summaries.

---

# 📊 Metrics Used

* Train/Test Accuracy
* Train/Test F1-score
* Best `k` for each model
* Comparison across fusion methods

---

# 🚀 Performance Optimization

* TensorFlow GPU memory growth
* EarlyStopping support
* Use stratified train–test splitting
* Experiment with additional feature  fusion and selection techniques
* Improve and optimize the model architectures

---

# 🧩 Extending the Project

You can easily add:

* Additional feature fusion methods
* Different feature selection techniques
* AutoML-based hyperparameter search
* Saving trained model weights for reuse
* Confusion matrix and classification reports inside the Word output
* Improved or alternative model architectures
* Additional deep learning or machine learning models

---

<!-- # 📜 License

MIT License — free for research & commercial use.

--- -->

<!-- # ⭐ Support

If this project helps you, please ⭐ star the repo!

--- -->

# 🙌 Contributions

<!-- Pull requests and suggestions are welcome. -->

This project is a group effort in collaboration with [CGhosh81](https://github.com/CGhosh81).

Contributors: `A3x-parvez`, `CGhosh81`
