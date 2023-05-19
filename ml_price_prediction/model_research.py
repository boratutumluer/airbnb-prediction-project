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
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
warnings.simplefilter(action='ignore', category=FutureWarning)

engine = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')
conn = engine.connect()

df_ = pd.read_sql_query("SELECT * FROM all_data", con=conn)
df = df_.copy()

columns = [col for col in df.columns if col not in ["index", "listing_url", "segment", "name", "host_has_profile_pic"]]
df = df[columns]

high_correlated_cols(df, plot=True)
########################################################################################################################
#                                       FEATURE ENGINEERING
########################################################################################################################
# rare_analyser(df, "price", ["property_type"])
tmp = df["property_type"].value_counts() / len(df)
labels = tmp[tmp < 0.05].index
df["property_type"] = np.where(df["property_type"].isin(labels), "Rare", df["property_type"])

df.columns = [col.upper().replace(" ", "_") for col in df.columns]

##############################################
#                   HISTOGRAMS
##############################################
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

# for col in num_cols:
#     num_summary(df, col, plot=True)
#
# for col in cat_cols:
#     cat_summary(df, col, plot=True)
#

#
# def boxplot_target_analysis_with_cat(dataframe, target, column):
#     df1 = dataframe.groupby(column)[target].mean().reset_index().sort_values(target, ascending=False)
#     plt.figure(figsize=(20, 15))
#     dft = dataframe[[column, target]].copy()
#     sns.boxplot(x=target, y=column, data=dft, order=df1[column].values,
#                 showfliers=False, palette="Spectral", linewidth=0.6, width=0.6)
#     ax = plt.gca()
#     ax.set_title("")
#     ax.set_xlabel(f"{target}", fontsize=12)
#     ax.set_ylabel("")
#     plt.suptitle(f"{target} by {column}", fontweight="bold", fontsize=16)
#
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label.set_fontsize(14)
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label.set_fontsize(14)
#     plt.savefig(
#         f"pythonProject/Github/Me/final_project/images_predict_price/exploratory/categorical/{column}_{target}.png",
#         bbox_inches='tight')

# for col in [col for col in cat_cols if df[col].nunique() >= 2]:
#     boxplot_target_analysis_with_cat(df, "PRICE", col)
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
#         f"pythonProject/Github/Me/final_project/images_predict_price/exploratory/numerical/{column}_{target}.png",
#         bbox_inches='tight')
#
#
# boxplot_target_analysis_with_num(df, "PRICE", "MINIMUM_NIGHTS")
# boxplot_target_analysis_with_num(df, "PRICE", "BATHROOMS")
# boxplot_target_analysis_with_num(df, "PRICE", "BEDS")
# boxplot_target_analysis_with_num(df, "PRICE", "BEDROOMS")
# boxplot_target_analysis_with_num(df, "PRICE", "ACCOMMODATES")

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

num_cols = [col for col in num_cols if col != "PRICE"]
# Normal dağılımın sağlanması için Log transformation uygulanması
for col in num_cols:
    df[col] = np.log1p(df[col])

sc = StandardScaler()
df[num_cols] = sc.fit_transform(df[num_cols])

binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)

ohe_cols = [col for col in cat_cols if 10 > df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df['NEIGHBOURHOOD_ENCODED'] = df.groupby('NEIGHBOURHOOD_CLEANSED')['PRICE'].transform('mean')
df['NEIGHBOURHOOD_ENCODED'] = np.log1p(df["NEIGHBOURHOOD_ENCODED"])
df["NEIGHBOURHOOD_ENCODED"] = sc.fit_transform(df[["NEIGHBOURHOOD_ENCODED"]])

########################################################################################################################
#                                                   BASE MODEL
########################################################################################################################
models = {"LGBM": LGBMRegressor(),
          "XGBoost": XGBRegressor(),
          "CatBoost": CatBoostRegressor(verbose=False)}

df["PRICE_LOG"] = np.log1p(df["PRICE"])
y = df["PRICE_LOG"]
X = df.drop(["PRICE", "PRICE_LOG", "NEIGHBOURHOOD_CLEANSED"], axis=1)
X.head()


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

cal_metric_for_regression(model, scoring=['neg_mean_squared_error', 'r2'], name=name)
'''
############## LGBM #################
Train RMSE:  0.3593
Test RMSE:  0.446
Train R2:  0.7348
Test R2:  0.571
############## XGBoost #################
Train RMSE:  0.2671
Test RMSE:  0.4604
Train R2:  0.8534
Test R2:  0.5425
############## CatBoost #################
Train RMSE:  0.3126
Test RMSE:  0.4397
Train R2:  0.7993
Test R2:  0.5825
'''
#######################################################################################################################
#                                                    MODEL TUNING
#######################################################################################################################
lgbm_model = LGBMRegressor()
lgbm_model.get_params()
lgbm_params = {"num-leaves": [15, 24, 31, 45, 60],
               "n_estimators": [100, 1000, 1500, 2000, 5000],
               "colsample_bytree": [0.3, 0.5, 0.7, 1],
               "feature_fraction": [0.5, 0.7, 0.8, 1],
               "bagging_fraction": [0.5, 0.7, 0.8, 1],
               "learning_rate": [0.1, 0.01, 0.001]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
final_lgbm_model = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)
cal_metric_for_regression(final_lgbm_model, scoring=['neg_mean_squared_error', 'r2'], name="LGBM")
'''
############## LGBM #################
Train RMSE:  0.3222
Test RMSE:  0.4385
Train R2:  0.7868
Test R2:  0.5852
'''

###################
# XgBoost
###################
xgboost_model = XGBRegressor()
xgboost_model.get_params()
xgboost_params = {"max_depth": [5, 6, 8, None],
                  "learning_rate": [0.001, 0.01, 0.1],
                  "colsample_bytree": [0.5, 0.8, 1, None],
                  "n_estimators": [100, 500, 1000]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_).fit(X, y)
cal_metric_for_regression(xgboost_final, scoring=['neg_mean_squared_error', 'r2'], name="XGBoost")

'''
############## XGBoost #################
Train RMSE:  0.272
Test RMSE:  0.4372
Train R2:  0.848
Test R2:  0.5881
'''

###################
# CatBoost
###################
catboost_model = CatBoostRegressor()
catboost_model.get_params()
catboost_params = {"iterations": [200, 500, 1000, 2000],
                   "depth": [3, 6, 10, 12],
                   "learning_rate": [0.1, 0.01, 0.001]}
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
final_catboost_model = catboost_model.set_params(**catboost_best_grid.best_params_).fit(X, y)
cal_metric_for_regression(final_catboost_model, scoring=['neg_mean_squared_error', 'r2'], name="CatBoost")


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


plot_importance(XGBRegressor().fit(X, y), X)


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


learning_curve_plot(xgboost_final, X, y)


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


val_curve_params(final_lgbm_model, X, y, "n_estimators", range(100, 5000), scoring="neg_root_mean_squared_error")
