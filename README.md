## Developed By : DODDA JAYASRI
## Register No : 212222240028
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

file_path = 'Salesforcehistory.csv'
data = pd.read_csv(file_path)
# Convert 'date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date to ensure time series continuity
data = data.sort_values('Date').set_index('Date')

# Plot the 'close' time series
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Close Price Time Series')
plt.show()

# Check stationarity of the 'close' time series
def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['Close'])

# Plot ACF and PACF
plot_acf(data['Close'])
plt.show()
plot_pacf(data['Close'])
plt.show()

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Define SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions on Stock Close Price')
plt.legend()
plt.show()
```


### OUTPUT:

![image](https://github.com/user-attachments/assets/52920093-d815-45dd-90b9-4410d5290de7)

![image](https://github.com/user-attachments/assets/0918327b-1d8a-47b4-9b4b-342efb05c3de)

![image](https://github.com/user-attachments/assets/4212284e-fe80-40b2-aedd-0b23c38f791a)


### RESULT:
Thus the program run successfully based on the SARIMA model.
