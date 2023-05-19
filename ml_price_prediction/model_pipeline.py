import numpy as np
import pandas as pd
import sys
import warnings
import joblib
from sqlalchemy import create_engine

sys.path.append('../airbnb/helpers')
from airbnb.helpers.data_prep import *
from airbnb.helpers.eda import *
from airbnb.helpers.pandas_options import set_pandas_options

pd.set_option("display.width", 220)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error

engine = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')
conn = engine.connect()

df = pd.read_sql_query("SELECT * FROM data_to_predict", con=conn)


def data_preprocessing(df):
    print("Data Preprocessing...")
    # Feature Engineering
    # Rare Analysis for PROPERTY_TYPE
    tmp = df["property_type"].value_counts() / len(df)
    labels = tmp[tmp < 0.05].index
    df["property_type"] = np.where(df["property_type"].isin(labels), "Rare", df["property_type"])

    from geopy.distance import great_circle
    def calculate_distance(latitude, longitude):
        istanbul_center = (41.0145545, 28.956243)
        listing = (latitude, longitude)
        return great_circle(istanbul_center, listing).km

    df['distance_from_center'] = df.apply(lambda x: calculate_distance(x.latitude, x.longitude), axis=1)

    district_count = df.groupby("neighbourhood_cleansed").size().reset_index().rename(columns={0: "count"})
    df = df.merge(district_count, on="neighbourhood_cleansed", how="left")
    df.index = df["id"]
    df.loc[(df["count"] >= 1000), "district_density"] = "very high density"
    df.loc[(1000 >= df["count"]) & (df["count"] > 500), "district_density"] = "high density"
    df.loc[(500 >= df["count"]) & (df["count"] > 200), "district_density"] = "middle density"
    df.loc[(200 >= df["count"]) & (df["count"] > 100), "district_density"] = "low density"
    df.loc[(df["count"] <= 100), "district_density"] = "very low density"
    df.drop("count", axis=1, inplace=True)

    df["host_response_rate"] = df["host_response_rate"] / 100

    df["host_segment"] = pd.qcut(df["host_since"], 5, ["E", "D", "C", "B", "A"])
    df["review_segment"] = pd.qcut(df["review_age"], 5, ["A,", "B", "C", "D", "E"])

    df.columns = [col.upper().replace(" ", "_") for col in df.columns]

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    num_cols = [col for col in num_cols if col != "PRICE"]

    # Log transformation
    for col in num_cols:
        df[col] = np.log1p(df[col])
    # Standardization
    sc = StandardScaler()
    df[num_cols] = sc.fit_transform(df[num_cols])
    # Label Encoding
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    # One-Hot Encoding
    ohe_cols = [col for col in cat_cols if 10 > df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)
    # Encoding for NEIGHBOURHOOD_CLEANSED
    df['NEIGHBOURHOOD_ENCODED'] = df.groupby('NEIGHBOURHOOD_CLEANSED')['PRICE'].transform('mean')
    df['NEIGHBOURHOOD_ENCODED'] = np.log1p(df["NEIGHBOURHOOD_ENCODED"])
    df["NEIGHBOURHOOD_ENCODED"] = sc.fit_transform(df[["NEIGHBOURHOOD_ENCODED"]])
    # Log transformation for PRICE
    df["PRICE_LOG"] = np.log1p(df["PRICE"])

    y = df["PRICE_LOG"]
    X = df.drop(["PRICE", "PRICE_LOG", "NEIGHBOURHOOD_CLEANSED", "ID"], axis=1)

    return X, y


def base_models(X, y, scoring=['neg_mean_squared_error', 'r2']):
    print("Base Models...")
    models = {"LGBM": LGBMRegressor(),
              "XGBoost": XGBRegressor(),
              "CatBoost": CatBoostRegressor(verbose=False)}

    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=True)
        train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'].mean())
        test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
        train_r2 = cv_results['train_r2'].mean()
        test_r2 = cv_results['test_r2'].mean()

        print(f"############## {name} #################")
        print("Train RMSE: ", round(train_rmse, 4))
        print("Test RMSE: ", round(test_rmse, 4))
        print("Train R2: ", round(train_r2, 4))
        print("Test R2: ", round(test_r2, 4))


# Hyperparameter Optimization
lgbm_params = {"colsample_bytree": [0.3, 0.5, 0.7, 1],
               "n_estimators": [2000, 5000, 7000, 8000],
               "learning_rate": [0.1, 0.01, 0.001]}

xgboost_params = {"max_depth": [8, 10, 12],
                  "learning_rate": [0.001, 0.01, 0.1],
                  "colsample_bytree": [0.3, 0.5, None],
                  "n_estimators": [1000, 3000]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

regressors = [("LGBM", LGBMRegressor(), lgbm_params),
              ("XGBoost", XGBRegressor(), xgboost_params),
              ("CatBoost", CatBoostRegressor(), catboost_params)]


def hyperparameter_optimization(X, y, cv=5, scoring=['neg_mean_squared_error', 'r2']):
    print("Hyperparameter Optimization...")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"############## {name} #################")
        cv_results = cross_validate(regressor, X, y, cv=10, scoring=scoring, return_train_score=True)
        print(f"Test R2 (Before): {round(cv_results['test_r2'].mean(), 4)}")
        print(f"Test RMSE (Before): {round(np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()), 4)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=1).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=10, scoring=scoring, return_train_score=True)
        print(f"Best Params: {gs_best.best_params_}")
        print(f"Test R2 (After): {round(cv_results['test_r2'].mean(), 4)}")
        print(f"Test RMSE (After): {round(np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()), 4)}")

        best_models[name] = final_model

    return best_models


def voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_reg = VotingRegressor(estimators=[("LGBM", best_models["LGBM"]),
                                             ("XGBoost", best_models["XGBoost"]),
                                             ("CatBoost", best_models["CatBoost"])]).fit(X, y)

    cv_results = cross_validate(voting_reg, X, y, cv=5, scoring=['neg_mean_squared_error', 'r2'],
                                return_train_score=True)
    train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'].mean())
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'].mean())
    train_r2 = cv_results['train_r2'].mean()
    test_r2 = cv_results['test_r2'].mean()

    print("############## Voted Regressor #################")
    print("Train RMSE: ", round(train_rmse, 4))
    print("Test RMSE: ", round(test_rmse, 4))
    print("Train R2: ", round(train_r2, 4))
    print("Test R2: ", round(test_r2, 4))

    return voting_reg


'''
{'colsample_bytree': 0.3, 'learning_rate': 0.01, 'n_estimators': 5000}
############## LGBM #################
Test R2 (After): 0.6093
Test RMSE (After): 0.4255


Best Params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 1000}
############## XGBoost #################
Test R2 (After): 0.6071
Test RMSE (After): 0.4271
'''
