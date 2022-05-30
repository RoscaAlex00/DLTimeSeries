import numpy
import pandas as pd
import numpy as np
import glob
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Loading the available data from the 'Alcohol' dataset
def load_alcohol():
    # Reading in all of the files from specific folders
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

    # Transform the raw dataset so that we have the time-lagged variables available as features (lag one)
    for i in range(len(list_raw)):
        list_raw[i] = prepare_data_alcohol(list_raw[i])
        list_raw[i]['start'] = pd.to_datetime(list_raw[i]['start'])
        # Add lagged variables
        lag_one = list_raw[i].shift()
        lag_one = lag_one.add_suffix('_1')
        list_raw[i] = pd.concat([list_raw[i], lag_one], axis=1)
        # Clean up the dataset so that we don't have entries with missing values
        list_raw[i] = list_raw[i][list_raw[i]['finish'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['finish_1'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['drinks'].notna()]
        list_raw[i] = list_raw[i][list_raw[i]['drinks_1'].notna()]
        # The lagged variables should be in the same day as the dependent feature
        list_raw[i] = list_raw[i][~(list_raw[i]['start'].dt.day != list_raw[i]['start_1'].dt.day)]

    # Concat for global model
    # train_df = pd.concat(df_train, ignore_index=True)
    # test_df = pd.concat(df_test, ignore_index=True)

    # train_df.drop(train_df.columns[0], axis=1, inplace=True)
    # test_df.drop(test_df.columns[0], axis=1, inplace=True)

    return df_train, df_test, list_raw


def prepare_data_alcohol(df_patient):
    # There is a patient that has a column which misses everywhere else so it's dropped
    if len(df_patient.columns) > 22:
        df_patient = df_patient.drop(columns=['Since the last survey; how many times have you used a tobacco product?'])
    # Replace the of variables as was done in the original study
    replacement_names = ['start', 'finish', 'drinks', 'comfortable', 'stressed', 'down', 'calm', 'pressure',
                         'enthusiastic', 'happy', 'conflict', 'craving', 'impulsive', 'pos_expect', 'peer_percent',
                         'want_drink', 'delay_grat', 'angry', 'drink_predict', 'restless_sleep', 'difficulty_sleep',
                         'hours_sleep']

    df_patient.columns = replacement_names

    return df_patient


# Function to load and pre-process the covid data
def load_covid():
    covid = pd.read_csv('data/covid/clean_ema.csv')

    # Replace with the names that were used in the study
    replacement_names = ["Relax", "Irritable", "Worry", "Nervous", "Future", "Anhedonia",
                         "Tired", "Hungry", "Alone", "Angry", "Social_offline", "Social_online", "Music",
                         "Procrastinate", "Outdoors", "C19_occupied", "C19_worry", "Home"]

    d = dict(zip(covid.columns[5:].values, replacement_names))

    # Clean-up
    covid = covid.rename(columns=d)
    covid = covid.drop(columns=['Scheduled', 'Issued', 'Response', 'Day'])
    covid['time'] = pd.to_datetime(covid['time'])

    # Label enconding of the ID's since the formatting in the original dataset is not suitable
    lbl = LabelEncoder()
    lbl.fit(list(covid['ID'].values))
    covid['ID'] = lbl.transform(list(covid['ID'].values))

    # Add lag one variables as we did for the alcohol dataset
    lag_one = covid.shift()
    lag_one = lag_one.add_suffix('_lag')
    covid = pd.concat([covid, lag_one], axis=1)
    covid = covid[covid['Duration'] != 'Expired']
    covid = covid[covid['Duration_lag'] != 'Expired']
    covid = covid[~(covid['time'].dt.day != covid['time_lag'].dt.day)]
    covid = covid[~(covid['ID'] != covid['ID_lag'])]

    # Remove participants that don't have at least 35 entries
    covid = covid[covid['ID'].map(covid['ID'].value_counts()) >= 35]
    covid = covid.dropna()
    print(len(np.unique(covid['ID'])))

    return covid


# Function to split the full COVID-19 dataset into smaller ones (per individual) and create train/test splits
def patients_covid():
    # Loading in the data from the above function
    covid_data = load_covid()

    # Split the dataset by ID's so that we have one for each participant
    id_list = covid_data['ID'].unique().tolist()
    covid_patients = []
    for i in id_list:
        df_patient = covid_data.loc[covid_data['ID'] == i]
        covid_patients.append(df_patient)

    # Print the ID's of patients that were included in the study
    print('Patient included in study:')
    print(id_list)

    covid_train_x_list, covid_test_x_list, covid_train_y_list, covid_test_y_list = [], [], [], []

    # Create train/test splits for every individual
    for j in range(len(covid_patients)):
        # Remove redundant variables
        current = covid_patients[j].drop(columns=['ID', 'ID_lag', 'time', 'time_lag', 'Duration', 'Duration_lag'])
        covid_X = current.drop(current.columns[range(0, 19)], axis=1)
        covid_y = current['C19_occupied']

        # Splitting into train/test
        # covid_train_x, covid_test_x, covid_train_y, covid_test_y = train_test_split(covid_X, covid_y, test_size=0.3,
        #                                                                            random_state=j)

        # Temporal splits
        test_size = int(0.3 * len(covid_X))

        covid_train_x = covid_X.iloc[:-test_size]
        covid_train_y = covid_y.iloc[:-test_size]

        covid_test_x = covid_X.iloc[-test_size:]
        covid_test_y = covid_y.iloc[-test_size:]

        covid_train_x_list.append(covid_train_x)
        covid_test_x_list.append(covid_test_x)
        covid_train_y_list.append(covid_train_y)
        covid_test_y_list.append(covid_test_y)

    return covid_train_x_list, covid_test_x_list, covid_train_y_list, covid_test_y_list


# Function that standardizes all of the columns in the dataset to have a mean = 0 and std = 1
def standardize(data):
    local = data.copy()
    for col in local.columns:
        local[col] = (local[col] - local[col].mean()) / np.std(local[col])
    return local


# Compute the metrics for a list of actual values and a list of predicted values
def eval_results(actual, predicted, show):
    # corr = np.corrcoef(predicted, actual)[0, 1]
    r2 = metrics.r2_score(actual, predicted)
    rmse = metrics.mean_squared_error(actual, predicted, squared=False)
    mae = metrics.mean_absolute_error(actual, predicted)
    mse = metrics.mean_squared_error(actual, predicted, squared=True)

    if show:
        print('R_squared:', r2)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        # print('CORR:', corr)

    return mse, rmse, mae


# Function that computes the average metrics, since there are too much information to display it per individual
def average_metrics(mse_list, rmse_list, mae_list):
    print('Average MSE:', np.mean(mse_list))
    print('Average RMSE:', np.mean(rmse_list))
    print('Average MAE:', np.mean(mae_list))
