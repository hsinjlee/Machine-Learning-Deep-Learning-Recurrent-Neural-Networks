import numpy as np
import pandas as pd 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#------------ FETCHING AND PREPROCESSING THE DATA ----------------

#we have a dataset for training and distinct dataset for testing
prices_dataset_train =  pd.read_csv('C:\\Users\\User\\Desktop\\SP500_train.csv')
prices_dataset_test =  pd.read_csv('C:\\Users\\User\\Desktop\\SP500_test.csv')

#we are after a given column in the dataset
trainingset = prices_dataset_train.iloc[:,5:6].values
testset = prices_dataset_test.iloc[:,5:6].values

#we use min-max normalization to normalize the dataset
min_max_scaler = MinMaxScaler(feature_range=(0,1))
scaled_trainingset = min_max_scaler.fit_transform(trainingset)

#we have to create the training dataset because the features are the previous values
#so we have n previous values: and we predict the next value in the time series
X_train = []
y_train = []

for i in range(40,1258):
    #0 is the column index because we have a single column
    #we use the previous 40 prices in order to forecast the next one
    X_train.append(scaled_trainingset[i-40:i,0]) 
    #indexes start with 0 so this is the target (the price tomorrow)
    y_train.append(scaled_trainingset[i,0])
 
#we want to handle numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

#input shape for LSTM architecture
#we have to reshape the dataset (numOfSamples,numOfFeatures,1)
#we have 1 because we want to predict the price tomorrow (so 1 value)
#numOfFeatures: the past prices we use as features
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#------------ BUILDING THE LSTM MODEL ----------------

#let's build the LSTM architecture
#return sequence true because we have another LSTM after this one
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.3))
model.add(Dense(units=1))

#RMSProp is working fine with LSTM but so do ADAM optimizer
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

#------------ TESTING THE ALGORITHM ----------------
#training set plus testset
dataset_total = pd.concat((prices_dataset_train['adj_close'],prices_dataset_test['adj_close']), axis=0) #vertical axis=0 horizontal axis=1
#all inputs for test set
inputs = dataset_total[len(dataset_total)-len(prices_dataset_test)-40:].values
inputs = inputs.reshape(-1,1)

#neural net trained on the scaled values we have to min-max normalize the inputs
#it is already fitted so we can use transform directly
inputs = min_max_scaler.transform(inputs)      

X_test = []

for i in range(40,len(prices_dataset_test)+40):
    X_test.append(inputs[i-40:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predictions = model.predict(X_test)

#inverse the predicitons because we applied normalization but we want to compare with the original prices
predictions = min_max_scaler.inverse_transform(predictions)

#plotting the results
plt.plot(testset, color='blue', label='Actual S&P500 Prices')
plt.plot(predictions, color='green', label='LSTM Predictions')
plt.title('S&P500 Predictions with Reccurent Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Epoch 1/100
# 39/39 [==============================] - 3s 76ms/step - loss: 0.0655
# Epoch 2/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0112
# Epoch 3/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0082
# Epoch 4/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0072
# Epoch 5/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0069
# Epoch 6/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0062
# Epoch 7/100
# 39/39 [==============================] - 3s 88ms/step - loss: 0.0062
# Epoch 8/100
# 39/39 [==============================] - 3s 89ms/step - loss: 0.0060
# Epoch 9/100
# 39/39 [==============================] - 4s 112ms/step - loss: 0.0059
# Epoch 10/100
# 39/39 [==============================] - 3s 79ms/step - loss: 0.0060
# Epoch 11/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0062
# Epoch 12/100
# 39/39 [==============================] - 3s 70ms/step - loss: 0.0062
# Epoch 13/100
# 39/39 [==============================] - 3s 71ms/step - loss: 0.0052
# Epoch 14/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0051
# Epoch 15/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0058
# Epoch 16/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0047
# Epoch 17/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0061
# Epoch 18/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0069
# Epoch 19/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0051
# Epoch 20/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0042
# Epoch 21/100
# 39/39 [==============================] - 3s 70ms/step - loss: 0.0044
# Epoch 22/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0045
# Epoch 23/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0039
# Epoch 24/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0045
# Epoch 25/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0053
# Epoch 26/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0044
# Epoch 27/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0043
# Epoch 28/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0040
# Epoch 29/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0044
# Epoch 30/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0038
# Epoch 31/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0039
# Epoch 32/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0038
# Epoch 33/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0034
# Epoch 34/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0036
# Epoch 35/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0042
# Epoch 36/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0037
# Epoch 37/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0034
# Epoch 38/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0036
# Epoch 39/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0033
# Epoch 40/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0034
# Epoch 41/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0035
# Epoch 42/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0038
# Epoch 43/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0031
# Epoch 44/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0033
# Epoch 45/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0032
# Epoch 46/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0031
# Epoch 47/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0029
# Epoch 48/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0030
# Epoch 49/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0028
# Epoch 50/100
# 39/39 [==============================] - 3s 70ms/step - loss: 0.0032
# Epoch 51/100
# 39/39 [==============================] - 3s 71ms/step - loss: 0.0034
# Epoch 52/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0031
# Epoch 53/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0029
# Epoch 54/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0029
# Epoch 55/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0027
# Epoch 56/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0028
# Epoch 57/100
# 39/39 [==============================] - 4s 91ms/step - loss: 0.0028
# Epoch 58/100
# 39/39 [==============================] - 4s 95ms/step - loss: 0.0024
# Epoch 59/100
# 39/39 [==============================] - 3s 84ms/step - loss: 0.0026
# Epoch 60/100
# 39/39 [==============================] - 3s 82ms/step - loss: 0.0026
# Epoch 61/100
# 39/39 [==============================] - 5s 120ms/step - loss: 0.0027
# Epoch 62/100
# 39/39 [==============================] - 3s 72ms/step - loss: 0.0026
# Epoch 63/100
# 39/39 [==============================] - 2s 63ms/step - loss: 0.0025
# Epoch 64/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0025
# Epoch 65/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0028
# Epoch 66/100
# 39/39 [==============================] - 3s 89ms/step - loss: 0.0026
# Epoch 67/100
# 39/39 [==============================] - 3s 76ms/step - loss: 0.0027
# Epoch 68/100
# 39/39 [==============================] - 4s 92ms/step - loss: 0.0023
# Epoch 69/100
# 39/39 [==============================] - 3s 86ms/step - loss: 0.0024
# Epoch 70/100
# 39/39 [==============================] - 2s 64ms/step - loss: 0.0023
# Epoch 71/100
# 39/39 [==============================] - 3s 68ms/step - loss: 0.0024
# Epoch 72/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0021
# Epoch 73/100
# 39/39 [==============================] - 2s 64ms/step - loss: 0.0026
# Epoch 74/100
# 39/39 [==============================] - 2s 63ms/step - loss: 0.0025
# Epoch 75/100
# 39/39 [==============================] - 2s 64ms/step - loss: 0.0027
# Epoch 76/100
# 39/39 [==============================] - 2s 63ms/step - loss: 0.0022
# Epoch 77/100
# 39/39 [==============================] - 2s 62ms/step - loss: 0.0021
# Epoch 78/100
# 39/39 [==============================] - 3s 78ms/step - loss: 0.0022
# Epoch 79/100
# 39/39 [==============================] - 3s 69ms/step - loss: 0.0021
# Epoch 80/100
# 39/39 [==============================] - 2s 61ms/step - loss: 0.0021
# Epoch 81/100
# 39/39 [==============================] - 2s 60ms/step - loss: 0.0021
# Epoch 82/100
# 39/39 [==============================] - 2s 60ms/step - loss: 0.0022
# Epoch 83/100
# 39/39 [==============================] - 2s 60ms/step - loss: 0.0022
# Epoch 84/100
# 39/39 [==============================] - 2s 63ms/step - loss: 0.0020
# Epoch 85/100
# 39/39 [==============================] - 3s 65ms/step - loss: 0.0021
# Epoch 86/100
# 39/39 [==============================] - 3s 74ms/step - loss: 0.0022
# Epoch 87/100
# 39/39 [==============================] - 3s 73ms/step - loss: 0.0020
# Epoch 88/100
# 39/39 [==============================] - 3s 88ms/step - loss: 0.0022
# Epoch 89/100
# 39/39 [==============================] - 3s 84ms/step - loss: 0.0019
# Epoch 90/100
# 39/39 [==============================] - 4s 110ms/step - loss: 0.0019
# Epoch 91/100
# 39/39 [==============================] - 3s 81ms/step - loss: 0.0019
# Epoch 92/100
# 39/39 [==============================] - 3s 88ms/step - loss: 0.0024
# Epoch 93/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0021
# Epoch 94/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0018
# Epoch 95/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0019
# Epoch 96/100
# 39/39 [==============================] - 3s 66ms/step - loss: 0.0019
# Epoch 97/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0018
# Epoch 98/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0018
# Epoch 99/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0017
# Epoch 100/100
# 39/39 [==============================] - 3s 67ms/step - loss: 0.0019
