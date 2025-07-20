# Diabetes Health Indicator Classification Project

**Course:** DS 675-102 – Machine Learning  
**Instructor:** Prof. Khalid Bakhshaliyev  
**Student:** [Your Name]

## 🧩 Problem Definition

### Task
Classify individual diabetes status into three categories:
- **0**: No diabetes
- **1**: Prediabetes
- **2**: Diabetes

### Dataset
- **Source**: BRFSS (Behavioral Risk Factor Surveillance System) 2015
- **Size**: 253,680 entries with 22 features
- **Motivation**: Early detection of diabetes for better health outcomes

### Goal
Build a high-accuracy classification model to predict diabetes status based on health indicators and demographic information.

## 📊 Dataset Details

**Health Conditions:** HighBP, HighChol, Stroke, Heart Disease  
**Lifestyle Indicators:** PhysActivity, Smoker, HvyAlcoholConsump, Fruits, Veggies  
**Healthcare Access:** AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk  
**Demographics:** Sex, Age, Education, Income  
**Target:** Diabetes_012 (0 = No diabetes, 1 = Prediabetes, 2 = Diabetes)

## 🔧 Pre-Processing Steps

1. Load the BRFSS 2015 dataset
2. Separate features from target variable
3. Handle class imbalance using SMOTE
4. Scale features with StandardScaler
5. Split data into training and test sets

## 📈 Exploratory Data Analysis

- Distribution of features
- Target class imbalance
- Correlation matrix (BMI, Age, General Health are strong predictors)

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC
- SHAP values for interpretability

## 🤖 Machine Learning Models

### Model 1: Deep Neural Network (DNN)
- Trained with ReLU, dropout, categorical cross-entropy
- Accuracy ≈ 85%
- Fails to detect Class 1 (Prediabetes)
- ROC-AUC for Class 1 was poor (~0.70)

### Model 2: Autoencoder Classifier
- Encoded features via autoencoder, then used a classifier
- Accuracy ≈ 64%
- Significantly better at detecting Class 1 and 2
- More balanced performance

## 🔍 SHAP Insights

Key features: BMI, General Health, HighBP, Smoking, Physical Activity

## ⚖️ Model Comparison

| Metric                | Deep Neural Network | Autoencoder Classifier |
|-----------------------|--------------------|-----------------------|
| Overall Accuracy      | ~85%               | ~64%                  |
| Class 0 (No Diabetes) | Excellent          | Good                  |
| Class 1 (Prediabetes) | Poor               | Good                  |
| Class 2 (Diabetes)    | Good               | Good                  |
| Class Balance         | Imbalanced         | More Balanced         |

## 🔧 Challenges & Future Work

- Class imbalance
- Overfitting
- Feature complexity

**Future Directions:**
- Try ensemble methods
- Advanced feature engineering
- Hyperparameter tuning
- Cross-validation
- Feature selection

## 📁 Project Structure

```
├── README.md                                    # Project documentation
├── diabetes_012_health_indicators_BRFSS2015.csv.zip  # Dataset
├── DNN_Code.ipynb                              # Deep Neural Network implementation
├── Autoencoder.ipynb                           # Autoencoder classifier implementation
├── presentations/                              # Folder for PPTX presentation files
└── pdfs/                                       # Folder for PDF files (reports, slides, etc.)
```

## 🚀 Getting Started

1. Download the dataset from the provided zip file
2. Open the Jupyter notebooks:
   - `DNN_Code.ipynb` for Deep Neural Network implementation
   - `Autoencoder.ipynb` for Autoencoder classifier implementation
3. Install required dependencies (see notebook cells)
4. Run the notebooks to reproduce results
5. Place your presentation slides in the `presentations/` folder and any PDF reports in the `pdfs/` folder

## 📚 References

- **Dataset**: [Kaggle BRFSS 2015](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Autoencoder Implementation**: Google Colab
- **DNN Implementation**: Google Colab
- **SMOTE**: [Imbalanced-learn library](https://imbalanced-learn.org/)
- **SHAP**: [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)

---

**Note:** This project demonstrates the trade-off between overall accuracy and balanced performance across all classes, highlighting the importance of choosing appropriate evaluation metrics based on the specific use case and business requirements. 