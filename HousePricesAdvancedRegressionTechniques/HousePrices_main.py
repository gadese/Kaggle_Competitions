import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test_modif.csv')

sale_price = train_dataset.iloc[:, -1].values
train_features = train_dataset.iloc[:, :-1].values
test_features = test_dataset.iloc[:, :].values


num_features_pos = [1, 3, 4, 17, 18, 19, 20, 26, 34, 36, 37, 38, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 59,
                    61, 62, 66, 67, 68, 69, 70, 71, 75, 76, 77]
features_one_hot = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33,
                    35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(train_features[:, num_features_pos])
# train_features[:, num_features_pos] = imputer.transform(train_features[:, num_features_pos])

#
# imputer = SimpleImputer(missing_values='NaN', strategy='mean')
# imputer.fit(train_features[:, num_features_pos])
train_features[:, num_features_pos] = imputer.transform(train_features[:, num_features_pos])
test_features[:, num_features_pos] = imputer.transform(test_features[:, num_features_pos])


X_train = np.zeros((train_features.shape[0], 1))
X_test = np.zeros((test_features.shape[0], 1))

for column in features_one_hot:
    labelencoder_features = LabelEncoder()
    temp = labelencoder_features.fit_transform(train_features[:, column].astype(str))
    temp_test = labelencoder_features.transform(test_features[:, column].astype(str))
    onehotencoder = OneHotEncoder(categories='auto')
    temp = onehotencoder.fit_transform(temp.reshape(temp.shape[0], 1)).toarray()
    temp_test = onehotencoder.transform(temp_test.reshape(temp_test.shape[0], 1)).toarray()
    X_train= np.append(X_train, temp, axis=1)
    X_test = np.append(X_test, temp_test, axis=1)

X_train = X_train[:, 1:]
X_test = X_test[:, 1:]


for column in num_features_pos:
    sc_X = StandardScaler()
    temp = sc_X.fit_transform(train_features[:, column].reshape(-1, 1).astype(float))
    temp_test = sc_X.transform(test_features[:, column].reshape(-1, 1).astype(float))
    X_train = np.append(X_train, temp, axis=1)
    X_test = np.append(X_test, temp_test, axis=1)


# sc_Y = StandardScaler()
# sale_price = sc_Y.fit_transform(sale_price.reshape(-1, 1))


model = MLPRegressor(hidden_layer_sizes= (100,50, 25, 10, 5,), activation= 'relu', solver= 'adam', learning_rate ='adaptive', max_iter = 1000, learning_rate_init=0.01, alpha=0.01)
model.fit(X_train, np.ravel(sale_price))
predicted_test = model.predict(X_test)

output = pd.DataFrame({'Id': test_dataset['Id'], 'SalePrice': predicted_test})
output.to_csv('prediction.csv', index=False)
