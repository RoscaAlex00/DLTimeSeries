import numpy
import pandas as pd
import numpy as np
import glob
import os
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
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

    covid = covid[covid['ID'].map(covid['ID'].value_counts()) >= 32]
    covid = covid.dropna()

    return covid


def patients_covid():
    covid_data = load_covid()

    id_list = covid_data['ID'].unique().tolist()
    covid_patients = []
    for i in id_list:
        df_patient = covid_data.loc[covid_data['ID'] == i]
        covid_patients.append(df_patient)

    print('Patient included in study:')
    print(id_list)

    covid_train_x_list, covid_test_x_list, covid_train_y_list, covid_test_y_list = [], [], [], []

    for j in range(len(covid_patients)):
        current = covid_patients[j].drop(columns=['ID', 'ID_lag', 'time', 'time_lag', 'Duration', 'Duration_lag'])
        covid_X = current.drop(current.columns[range(0, 19)], axis=1)
        covid_y = current['Worry']

        covid_train_x, covid_test_x, covid_train_y, covid_test_y = train_test_split(covid_X, covid_y, test_size=0.3,
                                                                                    random_state=j)
        covid_train_x_list.append(covid_train_x)
        covid_test_x_list.append(covid_test_x)
        covid_train_y_list.append(covid_train_y)
        covid_test_y_list.append(covid_test_y)

    return covid_train_x_list, covid_test_x_list, covid_train_y_list, covid_test_y_list


def standardize(data):
    local = data.copy()
    for col in local.columns:
        local[col] = (local[col] - local[col].mean()) / np.std(local[col])
    return local


def eval_results(actual, predicted, show):
    corr = np.corrcoef(predicted, actual)[0, 1]
    r2 = metrics.r2_score(actual, predicted)
    rmse = metrics.mean_squared_error(actual, predicted, squared=False)
    mae = metrics.mean_absolute_error(actual, predicted)

    if show:
        print('R_squared:', r2)
        print('MAPE:', metrics.mean_absolute_percentage_error(actual, predicted))
        print('RMSE:', rmse)
        print('MAE:', mae)
        print('CORR:', corr)

    return r2, rmse, mae


def eval_results_covid(actual, predicted, show):
    rmse = metrics.mean_squared_error(actual, predicted, squared=False)
    mae = metrics.mean_absolute_error(actual, predicted)
    mape = metrics.mean_absolute_percentage_error(actual, predicted)

    if show:
        print('MAPE:', mape)
        print('RMSE:', rmse)
        print('MAE:', mae)

    return mape, rmse, mae


def average_metrics(r2_list, rmse_list, mae_list):
    print('Average R_Squared:', np.mean(r2_list))
    print('Average RMSE:', np.mean(rmse_list))
    print('Average MAE:', np.mean(mae_list))


def average_metrics_covid(mape_list, rmse_list, mae_list):
    print('Average MAPE:', np.mean(mape_list))
    print('Average RMSE:', np.mean(rmse_list))
    print('Average MAE:', np.mean(mae_list))
