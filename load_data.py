import pandas as pd
import numpy as np
import glob
import os
from IPython.display import display
from sklearn.preprocessing import LabelEncoder


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

    for i in range(len(list_raw)):
        list_raw[i] = prepare_data_alcohol(list_raw[i])
        list_raw[i]['start'] = pd.to_datetime(list_raw[i]['start'])
        lag_one = list_raw[i].shift()
        lag_one = lag_one.add_suffix('_1')
        list_raw[i] = pd.concat([list_raw[i], lag_one], axis=1)
        list_raw[i] = list_raw[i][list_raw[i]['finish'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['finish_1'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['drinks'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['drinks_1'].notna()]
        list_raw[i] = list_raw[i][~(list_raw[i]['start'].dt.day != list_raw[i]['start_1'].dt.day)]

    # Concat for global model
    # train_df = pd.concat(df_train, ignore_index=True)
    # test_df = pd.concat(df_test, ignore_index=True)

    # train_df.drop(train_df.columns[0], axis=1, inplace=True)
    # test_df.drop(test_df.columns[0], axis=1, inplace=True)

    return df_train, df_test, list_raw


def prepare_data_alcohol(df_patient):
    if len(df_patient.columns) > 22:
        df_patient = df_patient.drop(columns=['Since the last survey; how many times have you used a tobacco product?'])
    # column names from paper
    replacement_names = ['start', 'finish', 'drinks', 'comfortable', 'stressed', 'down', 'calm', 'pressure',
                         'enthusiastic', 'happy', 'conflict', 'craving', 'impulsive', 'pos_expect', 'peer_percent',
                         'want_drink', 'delay_grat', 'angry', 'drink_predict', 'restless_sleep', 'difficulty_sleep',
                         'hours_sleep']

    df_patient.columns = replacement_names

    return df_patient


def load_covid():
    covid = pd.read_csv('data/covid/clean_ema.csv')

    replacement_names = ["Relax", "Irritable", "Worry", "Nervous", "Future", "Anhedonia",
                         "Tired", "Hungry", "Alone", "Angry", "Social_offline", "Social_online", "Music",
                         "Procrastinate", "Outdoors", "C19_occupied", "C19_worry", "Home"]

    d = dict(zip(covid.columns[5:].values, replacement_names))

    covid = covid.rename(columns=d)
    covid = covid.drop(columns=['Scheduled', 'Issued', 'Response', 'Day'])
    covid['time'] = pd.to_datetime(covid['time'])

    lbl = LabelEncoder()
    lbl.fit(list(covid['ID'].values))
    covid['ID'] = lbl.transform(list(covid['ID'].values))

    lag_one = covid.shift()
    lag_one = lag_one.add_suffix('_lag')
    covid = pd.concat([covid, lag_one], axis=1)
    covid = covid[covid['Duration'] != 'Expired']
    covid = covid[covid['Duration_lag'] != 'Expired']
    covid = covid[~(covid['time'].dt.day != covid['time_lag'].dt.day)]
    covid = covid[~(covid['ID'] != covid['ID_lag'])]

    return covid


