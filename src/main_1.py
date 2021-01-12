import matplotlib.pyplot as pyplot
import numpy as np
import sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
# -----------
# Read in csv data file
# Expected cols:  ['frame','roll','pitch','yaw','glance_location']
# -----------
def read_data(file_path):
    df = pd.read_csv(file_path, index_col=None)
    return df

class LabelTransFormer(BaseEstimator, TransformerMixin):
    def __init__(self, label_col):
        self.label_col = label_col
        self.le = LabelEncoder()
        self.is_fit=False
        return

    def fit(self, X, y=None):
        self.le.fit(X[self.label_col].values)
        self.is_fit =True
        return

    def fit_transform(self, X, y=None):
        if not self.is_fit:
            self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        x = self.le.transform(X[self.label_col].values)
        X.loc[:,self.label_col] = x
        return X

# -----------------------------
# Get a classifier object ( Pipeline to include preprocessing )
# Input classifier type:
# svm SVM
# lr Logistic regression
# gnb Gaussian Naive Bayes
# adaboost Ada Boost
# gbc Gradient Boosting
# -----------------------------
def create_classifier(_type='svm'):
    if _type == 'svm':
        clf_obj = SVC()
    elif _type == 'lr':
        clf_obj = LogisticRegression()
    elif _type == 'gnb':
        clf_obj = GaussianNB()
    elif _type == 'adaboost':
        clf_obj = AdaBoostClassifier()
    elif _type == 'gbc':
        clf_obj = GradientBoostingClassifier()

    clf_pipeline_object = Pipeline(
        steps=(
            ('preprocess_1',StandardScaler()),
            ('classifier',clf_obj)
        )
    )
    return clf_pipeline_object

def preprocess_data(
        data_df,
        y_column='y',
        shuffle = True,
        test_ratio = 0.2
):
    if shuffle:
        data_df = data_df.sample(frac=1.0)
    lt_obj = LabelTransFormer(y_column)
    data_df = lt_obj.fit_transform(data_df)
    train_df, test_df = train_test_split(data_df, test_size = test_ratio)
    return data_df, train_df, test_df

data_df = read_data()
data_df, train_df, test_df = preprocess_data(data_df)
clf_obj = create_classifier(_type='LR')
