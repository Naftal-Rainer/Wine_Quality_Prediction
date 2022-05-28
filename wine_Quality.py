import numpy as np 
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('WineQuality.csv')

dataset = dataset.dropna()

X = dataset.drop('quality', axis=1)

y = dataset['quality'].apply(lambda x: 1 if x >= 7 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Training
    
model = RandomForestClassifier()

model.fit(X_train,y_train)

# Accuracy on test data

x_test_prediction = model.predict(X_test)

training_data_accuracy = accuracy_score(x_test_prediction, y_test)

# Building a Predictive System

# input_data = (6.8,0.2,0.59,0.9,0.147,38.0,132.0,0.993,0,0.38,9.1)

def model_data(input_data):
    # Change to numpy array
    input_data_array = np.asarray(input_data)

    # Reshape the data
    input_data_reshaped = input_data_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    
    return prediction

# prediction = model_data(input_data)

# if (prediction[0] == 1):
#     print('Good Quality Wine')
# else:
#     print('Bad Quality Wine')




