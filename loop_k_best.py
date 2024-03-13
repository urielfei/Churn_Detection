import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from functions import *
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

## Read Data
df_raw = pd.read_csv('churn_data.csv')
## Clean Data
df_clean = clean_data(df_raw)
## Add New features/ Feature Enginering
df = features_eng(df_clean)

## X and y
X = df.drop(columns='Churn')
y = df['Churn']

## Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### Pipline

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

k_range = range(1, 35)
avg_scores = []
for k in k_range:
    # model = RandomForestClassifier(random_state=42)
    # params = {
    # 'max_depth': [3,5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    # 'max_features': ['sqrt'],
    # 'min_samples_leaf': [1, 2, 4],
    # 'min_samples_split': [2, 5, 10],
    # 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    model = LogisticRegression(random_state=42)

    pipeline = make_pipeline_imblearn(
        preprocessor,
        SelectKBest(f_classif, k=k),
        # SelectFromModel(xgboost.XGBClassifier()),
        SMOTE(sampling_strategy=0.25),
        model
    )
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)  # cv=5 for 5-fold cross-validation
    avg_scores.append(scores.mean())


plt.plot(k_range, avg_scores)
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection with Logistic Regression Model')
plt.show()

# Find the optimal k with the highest average score
optimal_k = k_range[avg_scores.index(max(avg_scores))]
print(f'The optimal number of features is: {optimal_k}')
# print(f'The optimal parameters are: {params}')

