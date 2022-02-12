import pandas as pd
import numpy as np
import glob
import os
from IPython.display import display


def load_alcohol():
    path_train = 'data/alcohol_data/train/*.csv'
    path_test = 'data/alcohol_data/test/*.csv'
    CURRENT_DIR = os.path.dirname(__file__)
    file_train = glob.glob(os.path.join(CURRENT_DIR, path_train))
    file_test = glob.glob(os.path.join(CURRENT_DIR, path_test))

    df_train = (pd.read_csv(file) for file in file_train)
    df_test = (pd.read_csv(file) for file in file_test)

    train_df = pd.concat(df_train, ignore_index=True)
    test_df = pd.concat(df_test, ignore_index=True)
    train_df.drop(train_df.columns[0], axis=1, inplace=True)
    test_df.drop(test_df.columns[0], axis=1, inplace=True)

    return train_df, test_df


def prepare_data_alcohol(df_patient):
    return -1


train, test = load_alcohol()

print(train.columns.values.tolist())
print(train.head(50))
