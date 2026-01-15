import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

"""### 
DATA COLLECTION AND PROCESSING
"""

#loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

#print(breast_cancer_dataset)

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#data_frame.head()

data_frame['label'] = breast_cancer_dataset.target

#data_frame.tail()

# getting some information about the data
#data_frame.info()

#checking the missing values

#data_frame.isnull().sum()

#statistical measures of data

#data_frame.describe()

data_frame['label'].value_counts()

"""0 --> Benign

1 --> Malignant
"""

data_frame.groupby('label').mean()

#separating the features and target

X = data_frame.drop(columns = 'label', axis = 1)
Y = data_frame['label']

"""### Spliting the data into training and testing"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

"""### Standardize the data

"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

"""## Importing tensorflow and keras"""

import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(30,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# training the neural network

history = model.fit(
    X_train_std,
    Y_train,
    epochs=30,
    validation_split=0.1,
    batch_size=32
)

"""### Accuracy of the model on test data"""

loss, accuracy = model.evaluate(X_test_std, Y_test)
print('Test accuracy:', accuracy)

Y_pred = model.predict(X_test_std)

"""#### model.predict() gives the prediction probability of each class for that data point"""

# ARGMAX FUNCTION
my_list = [10, 20, 30]

index_of_max = np.argmax(my_list)
print(index_of_max)

# converting the prediction probability for class labels

"""## PREDICTIVE MODEL"""

# To make a prediction with your own data, define input_data as a numpy array with 30 values.
# Example: input_data = np.array([17.99, 10.38, ..., 0.11890])
# For demonstration, we'll continue using the first row of X:
user_input_string = "849014,M,19.81,22.15,130,1260,0.09831,0.1027,0.1479,0.09498,0.1582,0.05395,0.7582,1.017,5.865,112.4,0.006494,0.01893,0.03391,0.01521,0.01356,0.001997,27.32,30.88,186.8,2398,0.1512,0.315,0.5372,0.2388,0.2768,0.07615"
input_data = np.array([float(x) for x in user_input_string.split(',')[2:]])

#change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standarizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

# sklearn breast cancer dataset uses:
# 0 = malignant, 1 = benign
# Sigmoid output is P(class=1) i.e., probability of BENIGN
if prediction[0][0] < 0.5:
  prediction_label = 0  # Malignant
else:
  prediction_label = 1  # Benign

print(prediction_label)

if prediction_label == 0:
  print('The breast cancer is Malignant')
else:
  print('The breast cancer is Benign')