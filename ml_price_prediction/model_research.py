import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import re
import datetime
import warnings
from sqlalchemy import create_engine

import sys

sys.path.append('../airbnb/helpers')
from airbnb.helpers.data_prep import *
from airbnb.helpers.eda import *

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, validation_curve, learning_curve, \
    train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import optuna

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
warnings.simplefilter(action='ignore', category=FutureWarning)

engine = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')
conn = engine.connect()

df_ = pd.read_sql_query("SELECT * FROM data_to_predict", con=conn)
df = df_.copy()


########################################################################################################################
#                                       FEATURE ENGINEERING
########################################################################################################################
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
df["review_segment"] = pd.qcut(df["review_age"], 5, ["A", "B", "C", "D", "E"])

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
##############################################
#                   HISTOGRAMS
##############################################


for col in num_cols:
    num_summary(df, col, plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

#
# def boxplot_target_analysis_with_cat(dataframe, target, column):
#     df1 = dataframe.groupby(column)[target].mean().reset_index().sort_values(target, ascending=False)
#     plt.figure(figsize=(20, 15))
#     dft = dataframe[[column, target]].copy()
#     sns.boxplot(x=target, y=column, data=dft, order=df1[column].values,
#                 showfliers=False, palette="Spectral", linewidth=0.6, width=0.6)
#     ax = plt.gca()
#     ax.set_title("")
#     ax.set_xlabel(f"{target}", fontsize=20)
#     ax.set_ylabel("")
#     plt.suptitle(f"{target} by {column}", fontweight="bold", fontsize=25)
#
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(20)
#     plt.savefig(f"graphs/{column}_{target}.png", bbox_inches='tight')
#
# for col in [col for col in cat_cols if df[col].nunique() >= 2]:
#     boxplot_target_analysis_with_cat(df, "PRICE", col)
#
# boxplot_target_analysis_with_cat(df, "PRICE", "NEIGHBOURHOOD_CLEANSED")
# boxplot_target_analysis_with_cat(df, "PRICE", "PROPERTY_TYPE")
#
#
# def boxplot_target_analysis_with_num(dataframe, target, column):
#     cov = np.round(dataframe[[column, target]].corr()[column][target], 3)
#     suptitle = f"{column} [Correlation: " + str(cov) + "]"
#
#     dft = dataframe[[column, target]].copy()
#     dft[column] = dft[column].astype(int)
#     dft.boxplot(by=column, showfliers=False, figsize=(20, 15), vert=False, patch_artist=True)
#
#     ax = plt.gca()
#     ax.set_title("")
#     ax.set_xlabel(f"{target}", fontsize=12)
#     ax.set_ylabel("")
#     plt.suptitle(suptitle, fontweight="bold", fontsize=12)
#
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(14)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(14)
#     plt.savefig(
#         f"graphs/{column}_{target}.png",bbox_inches='tight')
#
#
# boxplot_target_analysis_with_num(df, "PRICE", "MINIMUM_NIGHTS")
# boxplot_target_analysis_with_num(df, "PRICE", "BATHROOMS")
# boxplot_target_analysis_with_num(df, "PRICE", "BEDS")
# boxplot_target_analysis_with_num(df, "PRICE", "BEDROOMS")
# boxplot_target_analysis_with_num(df, "PRICE", "ACCOMMODATES")
# boxplot_target_analysis_with_num(df, "PRICE", "AMENITIES")

#################################################################
#                   SKEW/SCALING/ENCODING
#################################################################
# from scipy import stats
#
# def check_skew(df_skew, column):
#     skew = stats.skew(df_skew[column])
#     skewtest = stats.skewtest(df_skew[column])
#     plt.title('Distribution of ' + column)
#     sns.distplot(df_skew[column], color="g")
#     print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
#     return
#
# plt.figure(figsize=(9, 9))
# for i, v in enumerate(num_cols):
#     plt.subplot(i + 1, 1, i + 1)
#     check_skew(df, v)
# plt.tight_layout()
# # plt.savefig('before_transform.png', format='png', dpi=1000)
# plt.show(block=True)
########################################################################################################################
#                                                   BASE MODEL
########################################################################################################################
models = {"LGBM": LGBMRegressor(),
          "XGBoost": XGBRegressor(),
          "CatBoost": CatBoostRegressor(verbose=False)}

# Log transformation for PRICE
df["PRICE_LOG"] = np.log1p(df["PRICE"])

df.head()

y = df["PRICE_LOG"]
X = df.drop(["PRICE", "PRICE_LOG", "NEIGHBOURHOOD_CLEANSED", "ID"], axis=1)


def cal_metric_for_regression(model, scoring, name):
    model = model
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

    return train_rmse, test_rmse


for name, model in models.items():
    cal_metric_for_regression(model, scoring=['neg_mean_squared_error', 'r2'], name=name)

'''
############## LGBM #################
Train RMSE:  0.3464
Test RMSE:  0.4381
Train R2:  0.7536
Test R2:  0.5864
############## XGBoost #################
Train RMSE:  0.2562
Test RMSE:  0.4494
Train R2:  0.8652
Test R2:  0.5643
############## CatBoost #################
Train RMSE:  0.3004
Test RMSE:  0.4297
Train R2:  0.8147
Test R2:  0.6014

'''


#######################################################################################################################
#                                                    MODEL TUNING
#######################################################################################################################
def lightgbm_optimization(trial):
    # Define the search space for hyperparameters
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 10)
    }

    # Split the data into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the LightGBM model with the current set of hyperparameters
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Return the evaluation metric value (to be minimized)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(lightgbm_optimization, n_trials=100)

# study.best_params
# study.best_value
lgbm_model = LGBMRegressor(num_leaves=90,
                           learning_rate=0.01,
                           feature_fraction=0.40,
                           bagging_fraction=0.80,
                           bagging_freq=4,
                           min_child_samples=40,
                           n_estimators=4500,
                           min_data_in_leaf=8,
                           max_depth=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = lgbm_model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_score(y_test, y_pred)

# print("R2 score: ", r2_score(y_test, y_pred) * 100)
# print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
#
# # Error
# error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': y_pred.flatten()})
# print(error_diff.head(5))
# #    Actual Values  Predicted Values
# # 0       7.438972          6.814264
# # 1       5.874931          5.853006
# # 2       6.717805          6.877841
# # 3       6.643790          7.273637
# # 4       7.969704          8.158024
# # Visualize the error
# df1 = error_diff.head(25)
# df1.plot(kind='bar', figsize=(10, 7))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show(block=True)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_model.fit(X, y), X)


def learning_curve_plot(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X=X, y=y, cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            scoring="neg_root_mean_squared_error")

    plt.plot(train_sizes, np.mean(-train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(-test_scores, axis=1), 'o-', color="g", label="Validation score")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend(loc="best")
    plt.show(block=True)


learning_curve_plot(lgbm_model, X, y)


def val_curve_params(model, X, y, param_name, param_range, scoring, cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(-train_score, axis=1)
    mean_test_score = np.mean(-test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

