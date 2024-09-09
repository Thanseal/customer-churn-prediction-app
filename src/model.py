import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

#Load the model
def load_model(model_path='model.pkl'):
    with open (model_path, 'rb') as file:
        model = pickle.load(file)
    return model
    
#Predict using loaded model
def predict(model, features):
    predictions = model.predict(features)
    return predictions