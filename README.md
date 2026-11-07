# Job Acceptance Prediction System

## Project Overview
The Job Acceptance Prediction System is a Machine Learning-based application designed to analyze candidate information and predict job placement outcomes.

It uses historical candidate data and an XGBoost classification model to estimate whether a candidate is likely to be placed or not placed.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- MySQL
- Streamlit
- Plotly

## Machine Learning
Multiple classification algorithms were evaluated:
- Logistic Regression
- Random Forest
- XGBoost
- HistGradientBoosting

The tuned XGBoost model achieved the best recorded test performance:

| Metric | Score |
|---|---:|
| Accuracy | 92.37% |
| Precision | 87.43% |
| Recall | 87.35% |
| F1 Score | 87.39% |
| ROC-AUC | 97.84% |

These are experimental results from the project dataset and are not a guarantee of real-world performance.

## Dashboard Features
1. Dashboard Overview
2. Candidate Analysis
3. Placement Insights
4. Candidate Explorer
5. Job Acceptance Prediction

## Project Structure
```text
Job-Acceptance-Prediction/
│
├── app.py
├── requirements.txt
├── cleaned_candidates.csv
├── saved_model/
│   ├── xgboost_model.json
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── imputer.pkl
│   ├── numerical_features.pkl
│   ├── categorical_features.pkl
│   └── feature_names.pkl
│
└── README.md
```

## How to Run Locally

1. Clone the repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Configure your MySQL database and import the candidate dataset into the `candidates` table.
4. Create `.streamlit/secrets.toml` with your local MySQL connection details:

```toml
[mysql]
host = "localhost"
user = "root"
password = "YOUR_MYSQL_PASSWORD"
database = "job_acceptance_db"
```

5. Run the application:

```bash
python -m streamlit run app.py
```

## Important Notes
- Never upload database passwords or other credentials.
- The prediction model provides estimates, not guaranteed placement outcomes.
- Candidate data should only be published when you have permission and it contains no confidential information.

## Author
Dinesh Veeraraj

B.E. Computer Science Engineering | 2026
