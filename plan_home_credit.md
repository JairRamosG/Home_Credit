# Home Credit Default Risk — Project Plan

## Objective

Build a production-grade portfolio project demonstrating statistical testing, XGBoost modeling, and API-based deployment. The project predicts loan default probability using the Kaggle Home Credit Default Risk dataset and serves predictions via FastAPI with an interactive frontend.

---

## 1. Dataset Overview

**Source:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)

**Problem:** Predict whether a loan applicant will default (binary classification).

**Key Tables:**

| Table | Description | Key Columns |
|---|---|---|
| `application_train.csv` | Main table — one row per loan application | `SK_ID_CURR`, `TARGET`, demographics, financials |
| `bureau.csv` | Previous credits from other institutions | `SK_ID_CURR`, `SK_ID_BUREAU`, `CREDIT_ACTIVE` |
| `bureau_balance.csv` | Monthly balance snapshots of bureau credits | `SK_ID_BUREAU`, `MONTHS_BALANCE`, `STATUS` |
| `previous_application.csv` | Previous applications at Home Credit | `SK_ID_CURR`, `SK_ID_PREV`, `NAME_CONTRACT_STATUS` |
| `installments_payments.csv` | Payment history for previous loans | `SK_ID_PREV`, `NUM_INSTALMENT_NUMBER`, `DAYS_INSTALMENT` |
| `credit_card_balance.csv` | Monthly balance snapshots of Home Credit credit cards | `SK_ID_PREV`, `AMT_BALANCE`, `CNT_INSTALMENT_MATURECNT` |
| `POS_CASH_balance.csv` | POS and cash loan balances | `SK_ID_PREV`, `MONTHS_BALANCE`, `CNT_INSTALMENT` |

**Target Distribution:** ~92% non-default (0), ~8% default (1) — severe class imbalance.

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | FastAPI | Serve predictions and explanations via REST API |
| **Frontend** | HTML + Tailwind CSS + Plotly.js | Interactive UI with charts |
| **Model** | XGBoost + SHAP | Classification and feature explanation |
| **Deployment** | Render (single service) | FastAPI serves API + static files |
| **Development** | Jupyter Notebooks | EDA, feature engineering, model experimentation |

---

## 3. Folder Structure

```
home-credit-default/
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_statistical_tests.ipynb      # Hypothesis testing
│   ├── 03_feature_engineering.ipynb    # Feature creation and selection
│   └── 04_model_training.ipynb         # XGBoost training and tuning
├── src/
│   ├── api/
│   │   ├── main.py                     # FastAPI application entry point
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── dependencies.py             # Model loading and shared state
│   └── ml/
│       ├── train.py                    # Training pipeline (reproducible)
│       ├── predict.py                  # Inference logic
│       └── explain.py                  # SHAP explanation generation
├── static/
│   ├── index.html                      # Main page
│   ├── css/
│   │   └── styles.css                  # Tailwind + custom styles
│   └── js/
│       ├── app.js                      # Main application logic
│       └── charts.js                   # Plotly chart components
├── models/                             # Saved model artifacts
│   ├── xgboost_model.pkl               # Trained XGBoost model
│   ├── scaler.pkl                      # Feature scaler (if used)
│   ├── feature_names.json              # Feature column order
│   └── shap_background.pkl             # SHAP background data
├── data/                               # Raw dataset files
├── tests/
│   ├── test_api.py                     # API endpoint tests
│   └── test_ml.py                      # Model inference tests
├── requirements.txt
├── Dockerfile                          # For Render deployment
├── render.yaml                         # Render service config
├── README.md
└── plan_home_credit.md                 # This file
```

---

## 4. Development Phases

### Phase 1: Exploratory Data Analysis (EDA)

**Notebook:** `01_eda.ipynb`

**Tasks:**
- Load all tables and verify row counts, missing values, data types
- Analyze target distribution and confirm class imbalance (~92/8 split)
- Profile demographics: age, gender, family status, income type
- Profile financials: income, credit amount, annuity, goods price
- Analyze credit history features: days employed, days registration, external sources
- Visualize missing value patterns (missingno matrix)
- Identify high-cardinality categoricals and rare categories
- Detect outliers using IQR and domain knowledge (negative days = time before application)
- Save processed insights for downstream notebooks

**Deliverable:** EDA summary with key distributions, missing patterns, and initial feature observations.

---

### Phase 2: Statistical Testing

**Notebook:** `02_statistical_tests.ipynb`

**Objective:** Validate which features have statistically significant relationships with the target.

**Tests to Apply:**

| Test | Feature Type | Purpose | When to Use |
|---|---|---|---|
| **Chi-Square** | Categorical vs Target | Test independence between category and default | `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`, `CODE_GENDER`, `NAME_HOUSING_TYPE` |
| **Independent t-test** | Continuous (2 groups) | Compare means between default/no-default | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_EMPLOYED` |
| **Mann-Whitney U** | Continuous (non-normal) | Non-parametric alternative to t-test | Features that fail Shapiro-Wilk normality test |
| **ANOVA** | Continuous (3+ groups) | Compare means across multiple categories | Income across education levels, credit across income types |
| **Kruskal-Wallis** | Continuous (non-normal, 3+ groups) | Non-parametric alternative to ANOVA | Same as ANOVA but for non-normal features |
| **Cramér's V** | Categorical vs Categorical | Measure association strength between categoricals | Correlations between categorical features (multicollinearity check) |
| **Point-Biserial** | Continuous vs Binary target | Correlation between continuous feature and binary target | Quick correlation check for all numeric features |
| **Z-test for proportions** | Binary vs Binary | Compare proportions across groups | Default rate by gender, by income type |

**Workflow:**
1. For each feature, determine type (continuous/categorical/binary)
2. Test normality (Shapiro-Wilk for small samples, Anderson-Darling for large)
3. Apply appropriate test based on feature type and normality result
4. Calculate effect sizes (Cohen's d, Cramér's V, odds ratios)
5. Correct for multiple comparisons (Bonferroni or Benjamini-Hochberg)
6. Rank features by statistical significance and effect size
7. Document findings: which features are confirmed drivers of default risk

**Deliverable:** Table of all tests with p-values, effect sizes, and significance flags. Summary of confirmed feature-target relationships.

---

### Phase 3: Feature Engineering

**Notebook:** `03_feature_engineering.ipynb`

**Tasks:**

**A. Table Aggregations (from secondary tables):**

| Source Table | Aggregation | New Feature |
|---|---|---|
| `bureau.csv` | COUNT, SUM, AVG | `bureau_credit_count`, `bureau_active_count`, `bureau_total_amount` |
| `bureau_balance.csv` | GROUP BY status, mean months | `bureau_balance_mean_months`, `bureau_status_dist_*` |
| `previous_application.csv` | COUNT, SUM, ratios | `prev_app_count`, `prev_approved_ratio`, `prev_refused_ratio` |
| `installments_payments.csv` | LATE_PAYMENTS, AVG_DELAY | `installments_late_count`, `installments_avg_days_late` |
| `credit_card_balance.csv` | AVG_BALANCE, MAX_UTILIZATION | `cc_avg_balance`, `cc_max_utilization` |
| `POS_CASH_balance.csv` | COMPLETED_RATIO | `pos_completed_ratio` |

**B. Domain-Specific Features:**

- `credit_to_income_ratio` = AMT_CREDIT / AMT_INCOME_TOTAL
- `annuity_to_income_ratio` = AMT_ANNUITY / AMT_INCOME_TOTAL
- `goods_to_credit_ratio` = AMT_GOODS_PRICE / AMT_CREDIT
- `employment_ratio` = DAYS_EMPLOYED / DAYS_BIRTH
- `credit_term` = AMT_ANNUITY / (AMT_CREDIT * 0.01) — rough months
- `income_per_family_member` = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
- `flag_doc_sum` = sum of all FLAG_DOCUMENT_* columns

**C. Handling Missing Values:**

- Numerical: median imputation (robust to outliers)
- Categorical: mode imputation or "Missing" category
- Features with >50% missing: drop or flag only

**D. Encoding:**

- Low cardinality (<10 categories): one-hot encoding
- High cardinality: target encoding or frequency encoding

**E. Class Imbalance Handling:**

- Scale positive class weight: `scale_pos_weight = count(negative) / count(positive)`
- Consider SMOTE for training (but not for final production — explain why)

**F. Feature Selection:**

- Remove features with >95% constant values
- Remove features with correlation > 0.95 (keep the one with higher statistical significance)
- Apply SelectKBest or permutation importance for initial filtering

**Deliverable:** Processed feature matrix saved as `data/processed/`, feature engineering pipeline code in `src/ml/train.py`.

---

### Phase 4: Model Training

**Notebook:** `04_model_training.ipynb`

**Tasks:**

**A. Data Split:**
- Train/validation/test split: 70/15/15
- Stratified split to preserve target ratio
- Time-based split if temporal features exist (preferable)

**B. XGBoost Training:**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=ratio,          # handle imbalance
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,                   # L1 regularization
    reg_lambda=1.0,                  # L2 regularization
    early_stopping_rounds=50,
    random_state=42
)
```

**C. Hyperparameter Tuning:**
- Use Optuna or GridSearchCV
- Tune: max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, reg_alpha, reg_lambda
- Optimize for AUC-ROC (primary) and F1 (secondary)

**D. Evaluation:**

| Metric | Target | Why |
|---|---|---|
| **AUC-ROC** | > 0.75 | Primary metric — ranking quality |
| **Precision** | High | Cost of false positive: rejecting good applicant |
| **Recall** | High | Cost of false negative: lending to defaulter |
| **F1-Score** | Balance | Harmonic mean when precision/recall trade off |
| **PR-AUC** | > 0.40 | Better than ROC for imbalanced datasets |
| **Confusion Matrix** | — | Visualize error types and business cost |

**E. Model Interpretation:**
- Feature importance (gain-based)
- SHAP summary plot (global interpretation)
- SHAP force plot (individual prediction explanation)
- Partial dependence plots for top features
- ICE plots for interaction effects

**F. Model Export:**
- Save model as `models/xgboost_model.pkl`
- Save feature names as `models/feature_names.json`
- Save SHAP background data as `models/shap_background.pkl`

**Deliverable:** Trained model, evaluation report, SHAP visualizations, model artifacts saved to `models/`.

---

### Phase 5: FastAPI Backend

**Files:** `src/api/main.py`, `src/api/schemas.py`, `src/api/dependencies.py`

**Endpoints:**

| Method | Path | Description | Response |
|---|---|---|---|
| `GET` | `/` | Serve the frontend HTML | HTML |
| `GET` | `/health` | Health check | `{"status": "ok"}` |
| `GET` | `/model/info` | Model metadata (features, metrics) | `ModelInfoResponse` |
| `POST` | `/predict` | Predict default probability | `PredictionResponse` |
| `POST` | `/predict/batch` | Predict multiple applications | `BatchPredictionResponse` |
| `POST` | `/explain` | SHAP explanation for a prediction | `ExplanationResponse` |
| `GET` | `/stats/distributions` | Feature distributions for charts | `DistributionsResponse` |
| `GET` | `/stats/metrics` | Model performance metrics | `MetricsResponse` |

**Request/Response Schemas (Pydantic):**

```python
# schemas.py

class ApplicationData(BaseModel):
    # Demographics
    CODE_GENDER: str           # "M" or "F"
    NAME_EDUCATION_TYPE: str   # e.g., "Secondary special"
    NAME_FAMILY_STATUS: str    # e.g., "Married"
    NAME_INCOME_TYPE: str      # e.g., "Working"
    DAYS_BIRTH: float          # negative = days before application
    DAYS_EMPLOYED: float       # negative = days before application
    CNT_CHILDREN: int
    CNT_FAM_MEMBERS: int

    # Financials
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float

    # Credit history
    REGION_POPULATION_RELATIVE: float
    FLAG_OWN_CAR: str          # "Y" or "N"
    FLAG_OWN_REALTY: str       # "Y" or "N"
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "CODE_GENDER": "M",
                "NAME_EDUCATION_TYPE": "Higher education",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_INCOME_TYPE": "Commercial associate",
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -3500,
                "CNT_CHILDREN": 1,
                "CNT_FAM_MEMBERS": 3,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 1350000.0,
                "AMT_ANNUITY": 45000.0,
                "AMT_GOODS_PRICE": 1350000.0,
                "REGION_POPULATION_RELATIVE": 0.0188,
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.4
            }
        }


class PredictionResponse(BaseModel):
    prediction: int            # 0 or 1
    probability: float         # 0.0 to 1.0
    risk_level: str            # "low", "medium", "high"
    decision: str              # "approved", "review", "denied"


class ExplanationResponse(BaseModel):
    prediction: int
    probability: float
    feature_contributions: dict  # feature_name -> SHAP value
    top_positive_factors: list   # features pushing toward default
    top_negative_factors: list   # features pushing toward non-default
    shap_plot_url: str          # URL to rendered SHAP plot
```

**Application Logic:**

```python
# main.py (simplified)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Home Credit Default Risk Predictor")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(data: ApplicationData):
    # 1. Transform input to feature vector
    # 2. Load model and predict
    # 3. Calculate SHAP values
    # 4. Return prediction + explanation
    ...

@app.post("/explain")
async def explain(data: ApplicationData):
    # 1. Generate SHAP force plot
    # 2. Return feature contributions
    ...
```

**Model Loading (dependencies.py):**

```python
import joblib
import shap
import numpy as np

class ModelContainer:
    _instance = None

    def __init__(self):
        self.model = joblib.load("models/xgboost_model.pkl")
        self.feature_names = json.load(open("models/feature_names.json"))
        self.explainer = shap.TreeExplainer(
            self.model,
            data=joblib.load("models/shap_background.pkl")
        )

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**Deliverable:** Working FastAPI backend with all endpoints, request validation, and model serving.

---

### Phase 6: Frontend

**Files:** `static/index.html`, `static/css/styles.css`, `static/js/app.js`, `static/js/charts.js`

**Page Layout:**

```
┌─────────────────────────────────────────────────────────────┐
│  Header: "Home Credit Default Risk Predictor"               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │                  │  │  Charts Section                  │  │
│  │  Application     │  │                                  │  │
│  │  Form            │  │  [SHAP Summary Plot]             │  │
│  │                  │  │  [Feature Importance]             │  │
│  │  - Gender        │  │  [ROC Curve]                     │  │
│  │  - Education     │  │  [Confusion Matrix]              │  │
│  │  - Income        │  │                                  │  │
│  │  - Credit        │  └─────────────────────────────────┘  │
│  │  - ...           │                                       │
│  │                  │  ┌─────────────────────────────────┐  │
│  │  [ Predict ]     │  │  Prediction Result               │  │
│  │                  │  │                                  │  │
│  └─────────────────┘  │  Risk: LOW / MEDIUM / HIGH        │  │
│                       │  Probability: 0.12                 │  │
│                       │  Decision: APPROVED                │  │
│                       │                                  │  │
│                       │  [SHAP Force Plot for this input] │  │
│                       └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Footer: "Built with FastAPI + XGBoost | GitHub"            │
└─────────────────────────────────────────────────────────────┘
```

**Frontend Components:**

| Component | Library | Description |
|---|---|---|
| Form inputs | Tailwind CSS | Styled form with validation |
| Risk gauge | Plotly.js gauge chart | Visual risk indicator (green/yellow/red) |
| SHAP waterfall | Plotly.js waterfall | Feature contributions for this prediction |
| Feature importance | Plotly.js bar chart | Global model importance |
| ROC curve | Plotly.js line chart | Model performance visualization |
| Confusion matrix | Plotly.js heatmap | Error distribution |

**Tech Choices:**
- **Tailwind CSS:** Quick, clean styling without writing custom CSS
- **Plotly.js:** Interactive charts, works natively with Python data
- **Vanilla JS:** No framework overhead — forms, fetch API, DOM manipulation

**Deliverable:** Responsive, interactive frontend that communicates with FastAPI endpoints.

---

### Phase 7: Testing

**Files:** `tests/test_api.py`, `tests/test_ml.py`

**Test Coverage:**

| Test Type | What to Test | Tool |
|---|---|---|
| Unit tests | Feature engineering transforms | pytest |
| Unit tests | Model prediction pipeline | pytest |
| Integration tests | API endpoints with valid/invalid data | pytest + httpx |
| Contract tests | Response schemas match Pydantic models | pydantic validation |
| Load tests | Model inference latency under load | locust (optional) |

**Key Test Cases:**
- POST /predict with valid data returns 200 + correct schema
- POST /predict with missing fields returns 422 validation error
- POST /predict with extreme values (negative income, etc.) handled gracefully
- GET /model/info returns model metadata
- POST /explain returns SHAP values for all features
- Model predictions are consistent (same input → same output)

**Deliverable:** Test suite with >80% coverage on critical paths.

---

### Phase 8: Deployment

**Platform:** Render

**Configuration:**

```yaml
# render.yaml
services:
  - type: web
    name: home-credit-predictor
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Pre-deployment Checklist:**
- [ ] Model artifacts committed to `models/` (or downloaded at startup)
- [ ] `requirements.txt` pinned with exact versions
- [ ] `Dockerfile` tested locally
- [ ] API health endpoint returns 200
- [ ] Frontend loads and communicates with API
- [ ] README.md updated with live URL

**Deliverable:** Live, publicly accessible application on Render.

---

### Phase 9: Documentation & Portfolio

**README.md Structure:**

```markdown
# Home Credit Default Risk Predictor

## Live Demo
[Link to deployed app]

## Overview
Brief description of the problem, approach, and results.

## Tech Stack
- FastAPI (backend)
- XGBoost + SHAP (modeling)
- Tailwind CSS + Plotly.js (frontend)
- Render (deployment)

## Statistical Analysis
Summary of hypothesis tests performed and findings.

## Model Performance
| Metric | Score |
|---|---|
| AUC-ROC | 0.78 |
| F1-Score | 0.45 |
| Precision | 0.52 |
| Recall | 0.39 |

## Key Insights
- Top 5 features driving default risk
- Business implications of the model

## How to Run Locally
Step-by-step instructions.

## Project Structure
Folder tree with descriptions.

## Author
Your name + LinkedIn + GitHub
```

**Portfolio Integration:**
- Link from main GitHub profile README
- Add to portfolio website with screenshot
- Write a short blog post explaining the statistical testing approach
- Include in LinkedIn project section

---

## 5. Statistical Tests Reference

| Test | Null Hypothesis (H₀) | When to Reject | Effect Size |
|---|---|---|---|
| Chi-Square | Feature and target are independent | p < 0.05 | Cramér's V |
| t-test | Means are equal between groups | p < 0.05 | Cohen's d |
| Mann-Whitney | Distributions are equal | p < 0.05 | Rank-biserial correlation |
| ANOVA | All group means are equal | p < 0.05 | Eta-squared |
| Kruskal-Wallis | All group distributions are equal | p < 0.05 | Epsilon-squared |
| Point-Biserial | No correlation with binary target | p < 0.05 | r (correlation) |

**Multiple Comparison Correction:**
- Bonferroni: α_adjusted = α / number_of_tests (conservative)
- Benjamini-Hochberg: controls false discovery rate (preferred for many tests)

---

## 6. Timeline Estimate

| Phase | Estimated Time | Dependencies |
|---|---|---|
| 1. EDA | 4-6 hours | None |
| 2. Statistical Tests | 3-4 hours | Phase 1 |
| 3. Feature Engineering | 6-8 hours | Phase 2 |
| 4. Model Training | 4-6 hours | Phase 3 |
| 5. FastAPI Backend | 4-6 hours | Phase 4 |
| 6. Frontend | 6-8 hours | Phase 5 |
| 7. Testing | 2-3 hours | Phases 5, 6 |
| 8. Deployment | 1-2 hours | Phase 7 |
| 9. Documentation | 2-3 hours | Phase 8 |
| **Total** | **32-46 hours** | |

---

## 7. Interview Preparation Notes

**Questions you should be able to answer after this project:**

1. "Why did you choose XGBoost over other algorithms?" — Tabular data performance, handles missing values, feature importance built-in
2. "How did you handle the class imbalance?" — scale_pos_weight, evaluation with AUC-ROC instead of accuracy, threshold tuning
3. "What statistical tests did you use and why?" — Chi-square for categorical, t-test/Mann-Whitney for continuous, based on normality
4. "How do you explain the model's predictions?" — SHAP values show feature contributions per prediction
5. "What's the business cost of false negatives vs false positives?" — False negative: lending to defaulter (direct loss). False positive: rejecting good applicant (opportunity cost)
6. "How would you deploy this in production?" — FastAPI containerized, monitored for data drift, retrained periodically
7. "What feature engineering did you do?" — Aggregated secondary tables, domain ratios, handling missing values
8. "How did you validate the model?" — Stratified split, AUC-ROC, PR-AUC, confusion matrix analysis
