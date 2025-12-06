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
├── train_ids.py                   # Main pipeline script
├── README.md                      # Documentation
├── CIC-IDS-2017_fin_capped.csv    # Your dataset (not included)
└── Outputs/
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

### 2. Install dependencies

```bash
pip install -r requirment.txt
```

GPU version:

```bash
pip install tensorflow-gpu
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
* Stratified splitting
* Only one feature scoring computation per method
* Efficient reshaping for DL models

---

# 🧩 Extending the Project

You can easily add:

* XGBoost or LightGBM
* More feature fusion methods
* AutoML hyperparameter search
* Saving trained model weights
* Confusion matrices in reports

---

# 📜 License

MIT License — free for research & commercial use.

---

# ⭐ Support

If this project helps you, please ⭐ star the repo!

---

# 🙌 Contributions

Pull requests and suggestions are welcome.
