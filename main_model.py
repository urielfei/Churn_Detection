import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from imblearn.over_sampling import SMOTE
from functions import clean_data,features_eng
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt
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

# model = LogisticRegression(random_state=42)
model = RandomForestClassifier(random_state=42)
# 'max_depth': [3,5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
# 'max_features': ['sqrt'],
# 'min_samples_leaf': [1, 2, 4],
# 'min_samples_split': [2, 5, 10],
# 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

pipeline = make_pipeline_imblearn(
    preprocessor,
    SelectKBest(f_classif, k=31),
    # SelectFromModel(xgboost.XGBClassifier()),
    SMOTE(sampling_strategy=0.25),
    model
)


## Fit, Predict once
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test).tolist()
y_pred = pd.Series(y_pred)

f1 = f1_score(y_test, y_pred,pos_label='1')
f1_macro = f1_score(y_test, y_pred,average='macro',pos_label='1')

print('f1 Score: {}'.format(f1))
print('Macro f1 Score: {}'.format(f1_macro))
print(confusion_matrix(y_test,y_pred))

y_test = y_test.reset_index(drop=True)

y_prob = pipeline.predict_proba(X_test)
df_output = pd.DataFrame({'y_pred': y_prob[:, 1]})
df_output['y_true'] = y_test.astype(int)

# df_output = df_output.astype(int)
df_output = df_output.dropna()
bins = [0,0.05,0.1,0.3,0.45,0.6,0.75,1]
# bins = [0,0.2,0.4,0.6,0.8,1]
labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5','Level 6','Level 7']
# num_levels = 5
df_output['level'] = pd.cut(df_output['y_pred'], bins=bins, labels=labels,include_lowest=True)
churn_rates = df_output.groupby('level')['y_true'].agg({'count','mean'}).reset_index()
print("Churn Rate per Level:")
print(churn_rates)
# sns.lineplot(data=churn_rates, x="level", y="mean", kind="box")
# plt.show()

churn_rates.plot(kind='bar',x='level',y='mean',legend=None)
# plt.legend('')
plt.xlabel('Probablilty to Churn level')
plt.ylabel('Churn %')
plt.title('Churn Tiers Prediction')
plt.xticks(rotation=45)
plt.show()


