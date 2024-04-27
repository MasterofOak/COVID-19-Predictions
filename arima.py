import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, mse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn import metrics
from pandas.plotting import autocorrelation_plot
from tabulate import tabulate
from pmdarima import auto_arima
warnings.filterwarnings('ignore')


def filterData():
    df = pd.read_csv('poland.csv', parse_dates=True)
    # Pick only needed columns
    f_df = df[['date', 'total_cases']]
    # Change columns names
    f_df.columns = ['Date', 'Total']
    # Drop all 0 values
    f_df = f_df.fillna(0)
    f_df = f_df.loc[(f_df != 0).any(axis=1)]
    # Filter data for date
    start_date = f_df['Date'].min()
    end_date = '2022-12-31'
    f_df = f_df.query(
        'Date >= @start_date and Date <= @end_date', inplace=False)
    f_df = f_df.set_index('Date')
    return f_df


def difData(df):
    # Diffirenciate Data
    df = df.diff().dropna()
    return df


def performADFullerTest(df):
    test = adfuller(df['Total'], autolag='AIC')
    print('ADF -> ', test[0])
    print('P-value -> ', test[1])
    print('N of Lags Used -> ', test[2])
    return test


def autoARIMAcalcualtions(df):

    calc = auto_arima(df['Total'], trace=True, suppress_warnings=False)
    calc.summary()


def plotCorrGraphs(df):
    df = difData(df)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    ax[0] = plot_acf(df['Total'], lags=200, auto_ylims=True,  ax=ax[0])
    ax[1] = plot_pacf(df['Total'], lags=200, auto_ylims=True, ax=ax[1])
    plt.savefig('./plots/Correlations.png')
    plt.show()
    autocorrelation_plot(df['Total'])
    plt.show()


def my_ARIMA(df, train_data, test_data, delta):
    l_df = df.copy()
    # Training and testing data

    model_test = ARIMA(train_data['Total'], order=(4, 1, 0))
    model_test = model_test.fit()
    print('Summary of fitting model -> \n', model_test.summary())

    # Making prediction using testing data
    start = len(train_data)
    end = len(train_data) + len(test_data)-1
    test_pred = model_test.predict(start=start, end=end, type='levels')
    l_df['Testing Predictions'] = round(test_pred, 0)
    print('Testing Predictions ->\n',
          l_df['Testing Predictions'].tail(30).to_markdown())
    model_evaluation(test_data['Total'], test_pred, 'ARIMA')

    # Making prediction using all data
    model_final = ARIMA(l_df['Total'], order=(4, 1, 0))
    model_final = model_final.fit()
    final_pred = model_final.predict(start=len(l_df), end=len(
        l_df)+delta, type='levels')

    new_dates = pd.date_range(
        start='2023-01-01', periods=delta+1, normalize=True, freq='D')
    final_pred.index = new_dates

    l_df = pd.concat([l_df, final_pred], ignore_index=False)
    l_df['Total'].iloc[-(delta+1):] = final_pred
    l_df['Total'] = round(l_df['Total'], 0)
    l_df = l_df.drop(columns=['predicted_mean'])
    l_df.index = pd.to_datetime(l_df.index, format='%Y-%m-%d')
    print(l_df['Total'].tail(31).to_markdown())
    return l_df


def auto_ARIMA(df, train_data, test_data, delta):
    l_df = df.copy()
    # Training and testing data
    model_test = ARIMA(l_df['Total'], order=(5, 2, 1))
    model_test = model_test.fit()
    print('Summary of fitting model -> \n', model_test.summary())

    # Making prediction using testing data

    start = len(train_data)
    end = len(train_data) + len(test_data)-1
    test_pred = model_test.predict(start=start, end=end, type='levels')
    l_df['Testing Predictions'] = round(test_pred, 0)
    print('Testing Predictions ->\n',
          l_df['Testing Predictions'].tail(30).to_markdown())
    model_evaluation(test_data['Total'], test_pred, 'ARIMA')

    # Making prediction using all data

    model_final = ARIMA(l_df['Total'], order=(5, 2, 1))
    model_final = model_final.fit()
    final_pred = model_final.predict(start=len(l_df), end=len(
        l_df)+delta, type='levels')

    new_dates = pd.date_range(
        start='2023-01-01', periods=delta+1, normalize=True, freq='D')
    final_pred.index = new_dates

    l_df = pd.concat([l_df, final_pred], ignore_index=False)
    l_df['Total'].iloc[-(delta+1):] = final_pred
    l_df['Total'] = round(l_df['Total'], 0)
    l_df = l_df.drop(columns=['predicted_mean'])
    l_df.index = pd.to_datetime(l_df.index, format='%Y-%m-%d')
    print(l_df['Total'].tail(30).to_markdown())
    return l_df


def plotPredictions(df, auto, my):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.plot(np.array(auto.index), np.array(
        auto['Total']).reshape(-1, 1), color='green', label='Auto ARIMA Model')
    plt.plot(np.array(my.index), np.array(
        my['Total']).reshape(-1, 1), color='red', label='My ARIMA Model')
    plt.plot(np.array(df.index),
             np.array(df['Total']).reshape(-1, 1), color='blue', label='Initial Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('./plots/Final_Results.png')
    # plt.tight_layout()
    plt.show()


def model_evaluation(y_true, y_pred, Model):

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f'Model Evaluation: {Model}')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}')
    print(f'corr is : {np.corrcoef(y_true, y_pred)[0, 1]}', end='\n\n')


df = filterData()

train_data = df.iloc[:-30]
print('Train -> ', len(train_data))
test_data = df.iloc[-30:]
print('Test -> ', len(test_data))

auto = auto_ARIMA(df, train_data, test_data, 180)
my = my_ARIMA(df, train_data, test_data, 180)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
plotPredictions(df, auto, my)
