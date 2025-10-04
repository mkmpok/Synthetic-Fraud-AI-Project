import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def load_data(path='data/creditcard.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_for_model(df):
    df = df.copy()
    scaler = StandardScaler()
    df[['Time','Amount']] = scaler.fit_transform(df[['Time','Amount']])
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)