import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = []
Y_train = []
for ii in range(20, 1258):
    X_train.append(training_set[ii-20:ii, 0])
    Y_train.append(training_set[ii, 0])


X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200)


data_training = pd.read_csv('Google_Stock_Price_Train.csv')
data_test = pd.read_csv('Google_Stock_Price_Test.csv')
testing_set = pd.read_csv('Google_Stock_Price_Test.csv')
testing_set = testing_set.iloc[:, 1:2].values

# Getting the predicted stocck of 2017

dataset_total = pd.concat((data_training['Open'], data_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(data_test) - 20:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []

for jj in range(20, 40):
    X_test.append(inputs[jj-20:jj, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock = regressor.predict(X_test)
predicted_stock = sc.inverse_transform(predicted_stock)

#inputs = testing_set 
#inputs = sc.transform(inputs)
#inputs = np.reshape(inputs, (20, 1, 1))
#predicted_stock_price = regressor.predict(inputs)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(testing_set, color = 'red', label = 'Real google Stock price')
plt.plot(predicted_stock, color = 'blue', label = 'Predicted google Stock price')
plt.title('Google stock price prediction ')
plt.xlabel('time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error, accuracy_score
rmse = math.sqrt(mean_squared_error(testing_set, predicted_stock))









