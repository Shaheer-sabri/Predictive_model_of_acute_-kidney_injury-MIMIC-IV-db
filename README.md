# Predictive Model of Acute Kidney Injury in Critically Ill Patients with Acute Pancreatitis

This project employs a machine learning approach to predict acute kidney injury (AKI) in critically ill patients suffering from acute pancreatitis. The analysis and model development leverage the MIMIC-IV database, a comprehensive dataset containing de-identified health-related data associated with patients admitted to the intensive care unit (ICU).

## Project Objectives

1. Develop a machine learning model to predict the occurrence of AKI in critically ill patients with acute pancreatitis.
2. Identify key clinical features that contribute to the prediction of AKI.
3. Evaluate the performance of different machine learning algorithms.

## Dataset

- **Source:** MIMIC-IV database (Medical Information Mart for Intensive Care, Version IV).
- **Access:** Restricted access requiring credentialed permission through the PhysioNet platform.
- **Data Scope:** De-identified patient data, including demographics, lab results, vitals, medication administration, and clinical interventions.

## Methodology

### 1. Data Preprocessing
- Data cleaning: Handling missing values and outliers.
- Feature selection: Identification of relevant features based on clinical importance.
- Data normalization and encoding.

### 2. Model Development
- Machine learning algorithms used:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting Machines (XGBoost, LightGBM)
  - Neural Networks
- Training-validation split and hyperparameter tuning using grid search or random search.

### 3. Model Evaluation
- Metrics:
  - Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
  - Precision, Recall, F1 Score
  - Calibration curves for probability estimates

### 4. Feature Importance Analysis
- Techniques:
  - SHAP (SHapley Additive exPlanations)
  - Permutation importance

## Results

- The best-performing model was [Insert Best Model Name], achieving an AUC-ROC of [Insert Value].
- Key predictors of AKI included [Insert Top Features].
- [Insert Summary of Any Clinical Implications or Insights].

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/username/aki-prediction.git
   cd aki-prediction
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```bash
   jupyter notebook
   ```
   Navigate to the main notebook or script file and execute cells.

## File Structure

- `data/`: Contains preprocessed datasets (excluded in `.gitignore` for compliance with MIMIC-IV restrictions).
- `notebooks/`: Jupyter notebooks detailing the analysis and model development.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `results/`: Outputs such as model performance metrics and visualizations.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Dependencies

- Python >= 3.8
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn, SHAP

## Ethical Considerations

- The MIMIC-IV dataset contains de-identified patient data. Access to this dataset is governed by ethical and regulatory requirements, including completion of training on human subjects research.
- This project complies with all terms of use for the MIMIC-IV database.

## Acknowledgments

- The MIMIC-IV database and the researchers who developed and maintain it.
- Open-source libraries and tools used in this project.

## Future Work

- Validation of the model on external datasets.
- Deployment of the model as a clinical decision support tool.
- Exploration of deep learning models for improved performance.

