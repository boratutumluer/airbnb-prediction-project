import pandas as pd


def outlier_threshold(dataframe, column, q1=0.25, q3=0.75):
    Q1 = dataframe[column].quantile(q1)
    Q3 = dataframe[column].quantile(q3)
    IQR = Q3 - Q1
    max_limit = Q3 + 1.5 * IQR
    min_limit = Q1 - 1.5 * IQR
    print(f"for {column} --> min. limit: {min_limit}, max. limit: {max_limit}")
    return min_limit, max_limit


def check_outliers(dataframe, column, q1=0.25, q3=0.75):
    min_limit, max_limit = outlier_threshold(dataframe, column, q1, q3)
    if dataframe[(dataframe[column] < min_limit) | (dataframe[column] > max_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, column, q1=0.25, q3=0.75, index=False):
    min_limit, max_limit = outlier_threshold(dataframe, column, q1, q3)
    dataframe_with_outliers = dataframe[(dataframe[column] < min_limit) | (dataframe[column] > max_limit)]
    print(dataframe_with_outliers)
    if index:
        outlier_index = dataframe[(dataframe[column] < min_limit) | (dataframe[column] > max_limit)].index
        return outlier_index


def replace_with_threshold(dataframe, column, q1=0.25, q3=0.75):
    min_limit, max_limit = outlier_threshold(dataframe, column, q1, q3)
    dataframe.loc[dataframe[column] < min_limit, column] = min_limit
    dataframe.loc[dataframe[column] > max_limit, column] = max_limit


def remove_outliers(dataframe, column, q1=0.25, q3=0.75):
    min_limit, max_limit = outlier_threshold(dataframe, column, q1, q3)
    dataframe_without_outliers = dataframe[~((dataframe[column] < min_limit) | (dataframe[column] > max_limit))]
    return dataframe_without_outliers


def missing_values_table(dataframe, na_name=False):
    import numpy as np
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def label_encoder(dataframe, binary_categorical_col):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    dataframe[binary_categorical_col] = le.fit_transform(dataframe[binary_categorical_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    import numpy as np
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
