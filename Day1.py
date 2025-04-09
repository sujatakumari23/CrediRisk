# ============================
# 🗓️ DAY 1: Initial Setup, Utilities & Data Loading
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import warnings
import time
import os

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ✅ Utility function to track time usage
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"\n⏱️ Time taken by '{func.__name__}': {end - start:.2f}s")
        return result
    return wrapper

# ✅ Print data info, columns, and descriptive stats
def describe_data(df):
    print("\n🔍 Data Info:")
    print(df.info())
    print("\n🧾 Data Description:")
    print(df.describe(include='all'))
    print("\n🆔 Column Names:", df.columns.tolist())

# ✅ Load CSV file into DataFrame
@timer
def load_data(path):
    df = pd.read_csv(path)
    print("\n✅ Data loaded successfully! Shape:", df.shape)
    return df
