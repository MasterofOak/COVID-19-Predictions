import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import metrics
from scipy.optimize import curve_fit

# Sample COVID-19 data (daily cases)
# Replace this with actual data


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

# Define the Gaussian function


def gaussian_function(t, a, b, c):
    return b * np.exp(-((t - a) ** 2) / (2 * c ** 2))


df = filterData()
days = np.arange(len(df))
data = np.array(df['Total'])
# Fit the Gaussian function to the data
# Initial guess for parameters: a=20, b=10, c=1000
train_data = df.iloc[:-30]
test_data = df.iloc[-30:]

popt, pcov = curve_fit(gaussian_function, np.arange(
    len(train_data)), np.array(train_data['Total']), p0=[2000, 100, 1000])

# Extract regression parameters
a, b, c = popt
# Fit the model

y_fit = gaussian_function(np.arange(len(train_data)), *popt)
train_future_days = np.arange(len(train_data), len(train_data)+30)
y_train_future = gaussian_function(train_future_days, *popt)
train_future_df = pd.DataFrame(y_train_future)
# Test the model
model_evaluation(test_data['Total'], train_future_df, 'Gaussian')

# Predict Future Cases
future_days = np.arange(len(df), len(df)+120)
y_future = gaussian_function(future_days, *popt)
future_df = pd.DataFrame(y_future)

print(future_df.to_markdown())

# Plot the data and the fitted Gaussian function

plt.scatter(days, data, label='Data')
plt.plot(np.arange(len(train_data)), y_fit,
         'r--', linewidth=4, label='Fitted Gaussian Function')
plt.plot(np.arange(len(df)), df['Total'],
         color='blue', linewidth=4, label='Validation Data')
plt.plot(future_days, np.array(y_future),
         'g--', linewidth=4, label='Predicted Cases')

plt.xlabel('Days')
plt.ylabel('COVID-19 Cases')
plt.legend()
plt.grid(True)
plt.title('Fitted Gaussian Function to COVID-19 Data')
plt.savefig('./plots/Gaussian.png')
plt.show()

print("Regression Parameters:")
print("a:", a)
print("b:", b)
print("c:", c)
print('Matrix:\n', pcov)
