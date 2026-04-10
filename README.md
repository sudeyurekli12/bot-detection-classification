# Bot vs. Human Classification — Social Media Dataset
 
**Course:** Introduction to Machine Learning — Tilburg University (2025)  
**Author:** Sude Yurekli (Individual Assignment)  
**Tools:** Python, scikit-learn, pandas, matplotlib
 
---
 
## The Problem
 
Can we automatically distinguish between automated bot accounts and genuine users on a social media platform, using only profile data?
 
This was an individual machine learning assignment built around a real-world binary classification problem. The dataset included numerical features (activity metrics, follower counts) and categorical features (profile photo presence, privacy settings), with the target variable being `1` for bots and `0` for genuine users.
 
---
 
## What I Did
 
### Data Preparation
- Dropped columns with more than 50% missing or unknown values
- Removed remaining rows with NaN values after cleaning
- Applied `OneHotEncoder` to handle categorical string features (e.g. city)
- Split data 80/20 into training and test sets (random state 42)
 
### Models
I trained and compared two models:
 
| Model | Role | Rationale |
|---|---|---|
| Logistic Regression | Baseline | Simple, interpretable, well-suited for binary classification |
| Random Forest | Main model | Handles mixed feature types, provides feature importance |
 
### Hyperparameter Tuning
Both models were tuned using **Grid Search with 5-fold cross-validation**:
- Logistic Regression: tuned `C`, `penalty`, `solver` → best: `C=10, penalty=l2`
- Random Forest: tuned `n_estimators`, `max_depth`, `min_samples_split` → best: `150 trees, max_depth=15`
 
---
 
## Results
 
| Model | Accuracy | F1 (bots) | AUC |
|---|---|---|---|
| Logistic Regression (tuned) | **94.9%** | **0.767** | — |
| Random Forest (tuned) | 90.8% | 0.340 | — |
 
Logistic Regression outperformed Random Forest overall, particularly on bot recall. Random Forest achieved perfect precision on bots (1.000) but very low recall (0.205), meaning it missed most bots — a significant weakness for this use case.
 
### Key Finding
Logistic Regression's feature coefficients and Random Forest's feature importance both pointed to **activity metrics and follower/following ratios** as the strongest signals for identifying bots.
 
---
 
## Honest Reflection
 
The dataset was imbalanced — only ~12% of accounts were bots — which hurt Random Forest's recall significantly. Techniques like SMOTE oversampling or class weighting could improve bot detection in future iterations. Logistic Regression handled the imbalance better due to its probabilistic nature.
 
---
 
## Files
- `ML_2_Code_Finished.ipynb` — full notebook with preprocessing, training, tuning, evaluation, and visualisations
