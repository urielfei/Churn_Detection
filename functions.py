import pandas as pd
import numpy as np
import xgboost
from scipy import stats


def clean_data(raw_df):
    # Duplicates
    out_df = raw_df.drop_duplicates()
    out_df = out_df.set_index('CustomerID')

    # Types
    c = ['PreferredLoginDevice', 'CityTier', 'SatisfactionScore', 'Complain', 'Churn','HourSpendOnApp']
    out_df[c] = out_df[c].astype(str)

    # Nulls
    # Imputations
    out_df['Tenure'] = np.where(out_df['Tenure'].isnull(), out_df['Tenure'].mode()[0], out_df['Tenure'])
    out_df['WarehouseToHome'] = out_df['WarehouseToHome'].fillna(0)
    out_df['HourSpendOnApp'] = out_df['HourSpendOnApp'].fillna('Other')
    out_df['OrderAmountHikeFromlastYear'] = out_df['OrderAmountHikeFromlastYear'].fillna(0)
    out_df['CouponUsed'] = out_df['CouponUsed'].fillna(0)
    out_df['OrderCount'] = out_df['OrderCount'].fillna(0)
    out_df['DaySinceLastOrder'] = out_df['DaySinceLastOrder'].fillna(0)

    # Outliers
    cols_numeric = out_df.select_dtypes(include=np.number).columns.tolist()
    for col in cols_numeric:
        z = np.abs(stats.zscore(out_df[col]))
        threshold = 3
        idx = z[z < threshold].index
        out_df = out_df.loc[idx]

    return out_df


def feature_selection(X_train, y_train, X_test, n_select_features=None):
    if n_select_features is None:
        n_select_features = X_train.shape[1]
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)

    feat_imp = model.feature_importances_
    feature_importance_df = pd.Series(feat_imp, index=X_train.columns)
    feature_importance_df = feature_importance_df.sort_values(ascending=False)
    feature_importance_df = feature_importance_df[0:n_select_features]
    # feature_importance_df.to_csv('feature_importance_log.csv',index=True)
    sorted_idx = np.argsort(feat_imp)[::-1]
    top_features = [X_train.columns[i] for i in sorted_idx[:n_select_features]]
    # print(top_features)

    X_train_new = X_train[top_features]
    X_test_new = X_test[top_features]
    return X_train_new, X_test_new, feature_importance_df


def features_eng(df):
    df_out = df.copy()

    df_out['NumberOfAddress'] = np.where(df_out['NumberOfAddress'] > 5, '6+', df_out['NumberOfAddress'])
    # df_out['HourSpendOnApp'] = pd.to_numeric(df_out['HourSpendOnApp'])
    # df_out['HourSpendOnApp'] = np.where(df_out['HourSpendOnApp']<3,'2-','+3')

    df_out['PreferedOrderCat'] = df_out['PreferedOrderCat'].astype(str)
    df_out['PreferedOrderCat'] = df_out['PreferedOrderCat'].apply(lambda x: 'mob' if 'mobile' in x.lower() else x)

    # df_out['CouponUsed_levels'] = np.where(df_out['CouponUsed'] > 2, 3, df_out['CouponUsed'])

    df_out['CouponUsed'] = pd.to_numeric(df_out['CouponUsed'])
    df_out['CouponUsed'] = df_out['CouponUsed'].fillna(0)
    df_out['OrderCount'] = pd.to_numeric(df_out['OrderCount'])
    df_out['pct_CouponUsed'] = (df_out['CouponUsed']/df_out['OrderCount'])
    df_out['pct_CouponUsed'] = df_out['pct_CouponUsed'].fillna(df_out['pct_CouponUsed'].mean())

    df_out = df_out.replace([np.inf, -np.inf], 0)

    cols_to_use = ['Churn', 'Tenure', 'CityTier',
                   'WarehouseToHome', 'PreferredPaymentMode','NumberOfAddress',
                   'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
                   'Complain', 'OrderAmountHikeFromlastYear', 'pct_CouponUsed', 'OrderCount',
                   'DaySinceLastOrder', 'CashbackAmount']

    # df_out['OrderAmountHikeFromlastYear'] = df_out['OrderAmountHikeFromlastYear'] / 100
    return df_out[cols_to_use]