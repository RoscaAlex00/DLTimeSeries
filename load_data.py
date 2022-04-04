import pandas as pd
import numpy as np
import glob
import os
from IPython.display import display


def load_alcohol():
    path_train = 'data/alcohol_data/train/*.csv'
    path_test = 'data/alcohol_data/test/*.csv'
    path_raw = 'data/alcohol_data/raw/*.csv'
    CURRENT_DIR = os.path.dirname(__file__)
    file_train = glob.glob(os.path.join(CURRENT_DIR, path_train))
    file_test = glob.glob(os.path.join(CURRENT_DIR, path_test))
    file_raw = glob.glob(os.path.join(CURRENT_DIR, path_raw))

    df_train = [pd.read_csv(file) for file in file_train]
    df_test = [pd.read_csv(file) for file in file_test]
    list_raw = [pd.read_csv(file) for file in file_raw]

    # Concat for global model
    # train_df = pd.concat(df_train, ignore_index=True)
    # test_df = pd.concat(df_test, ignore_index=True)

    # train_df.drop(train_df.columns[0], axis=1, inplace=True)
    # test_df.drop(test_df.columns[0], axis=1, inplace=True)

    return df_train, df_test, list_raw


def prepare_data_alcohol(df_patient):
    if len(df_patient.columns) > 22:
        df_patient = df_patient.drop(columns=['Since the last survey; how many times have you used a tobacco product?'])
    replacement_names = ['start', 'finish', 'drinks', 'comfortable', 'stressed', 'down', 'calm', 'pressure',
                         'enthusiastic', 'happy', 'conflict', 'craving', 'impulsive', 'pos_expect', 'peer_percent',
                         'want_drink', 'delay_grat', 'angry', 'drink_predict', 'restless_sleep', 'difficulty_sleep',
                         'hours_sleep']

    df_patient.columns = replacement_names
    df_patient['start'] = pd.to_datetime(df_patient['start'])
    df_patient = df_patient.set_index('start')
    df_patient = df_patient.loc[df_patient['finish'].notnull()]
    df_patient['finish'] = pd.to_datetime(df_patient['finish'])


    return df_patient
