import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV


def fillna_knn(
    df: pd.DataFrame,
    base: List | np.ndarray,
    target: str,
    tuning_params: dict,
    n_jobs=-1,
):
    print("Target: ", target)
    whole = [target] + base

    miss = df[target].isna()
    notmiss = ~miss

    X_target = df.loc[notmiss, whole]

    X = X_target[base]
    Y = X_target[target]

    X_train_80, X_test, y_train_80, y_test = train_test_split(X, Y, test_size=0.20)

    # print('Fitting...')
    model = KNeighborsClassifier(weights="uniform")
    tuned_model = GridSearchCV(model, tuning_params, cv=5, verbose=0, n_jobs=n_jobs)
    tuned_model.fit(X_train_80, y_train_80)

    print("Best Score: {:.3f}".format(tuned_model.best_score_))
    print("Best Params: ", tuned_model.best_params_)
    test_acc = accuracy_score(y_true=y_test, y_pred=tuned_model.predict(X_test))
    print("Test Accuracy: {:.3f}".format(test_acc))

    # print('Predicting missing values...')
    Z = tuned_model.predict(df.loc[miss, base])

    df.loc[miss, target] = Z
    print("Done!\n")


# Function to deal with variables that are actually string/categories
def zoningcode2int(df: pd.DataFrame, target: str):
    print("Target: ", target)
    print("Dealing with variables that are actually string/categories...")
    storenull = df[target].isna()
    enc = LabelEncoder()
    df[target] = df[target].astype(str)

    # print('fit and transform')
    df[target] = enc.fit_transform(df[target])
    print("num of categories: ", enc.classes_.shape)
    df.loc[storenull, target] = np.nan
    # print('recover the nan value\n')
    return enc


# Slightly amend the k nearest neighbour function so it works on regression
def fillna_knn_reg(
    df: pd.DataFrame,
    base: List | np.ndarray,
    target: str,
    tuning_params: dict,
    n_jobs=-1,
):
    print("Target: ", target)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(df[base].values)
    rescaledX = scaler.transform(df[base].values)

    X = rescaledX[df[target].notnull()]
    Y = df.loc[df[target].notnull(), target]

    X_train_80, X_test, y_train_80, y_test = train_test_split(X, Y, test_size=0.20)

    # print('Fitting...')
    model = KNeighborsRegressor()
    tuned_model = GridSearchCV(
        model,
        tuning_params,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=0,
        n_jobs=n_jobs,
    )
    tuned_model.fit(X_train_80, y_train_80)

    print("Best Score: {:.4f}".format(-tuned_model.best_score_))
    print("Best Params: ", tuned_model.best_params_)
    test_mse = root_mean_squared_error(
        y_true=y_test, y_pred=tuned_model.predict(X_test)
    )
    print("MSE: {:.4f}".format(test_mse))

    # print('Predicting missing values...')
    Z = tuned_model.predict(rescaledX[df[target].isnull()])
    df.loc[df[target].isnull(), target] = Z
    print("Done!\n")


def plot_feature_importance(importance: List | np.ndarray, names, model_type, limit=35):
    # Create arrays from feature importance and feature names
    feature_importance = np.asarray(importance)
    feature_names = np.asarray(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 8))
    # Plot Searborn bar chart
    sns.barplot(
        x=fi_df["feature_importance"].iloc[:limit],
        y=fi_df["feature_names"].iloc[:limit],
    )
    # Add chart labels
    plt.title(model_type)
    plt.xlabel("FEATURE IMPORTANCE")
    plt.ylabel("FEATURE NAMES")


def to_float64_float32(df: pd.DataFrame):
    float_cols = df.select_dtypes(np.float64)
    # Reduce occupied memory by 600MB
    df[float_cols.columns] = float_cols.astype(np.float32)


def to_int64_int32(df: pd.DataFrame):
    int_cols = df.select_dtypes(np.int64)
    # Reduce occupied memory by 600MB
    df[int_cols.columns] = int_cols.astype(np.int32)
