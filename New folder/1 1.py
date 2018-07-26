import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
Y_train = training_set[1:1258]

X_train = np.reshape(X_train, (1257, 1, 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM

regressor = Sequential()
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200)

testing_set = pd.read_csv('Google_Stock_Price_Test.csv')
testing_set = testing_set.iloc[:, 1:2].values

# Getting the predicted stocck of 2017

inputs = testing_set 
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(testing_set, color = 'red', label = 'Real google Stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted google Stock price')
plt.title('Google stock price prediction ')
plt.xlabel('time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()