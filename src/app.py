from flask import Flask, request, jsonify
import pandas as pd
from model import load_model, predict

app = Flask(__name__)

#Load the model
model = load_model('model.pk1')

@app.route('/predict', methods=['POST'])
def predict_churn():
    #Get data from JSON
    data = request.json
    df = pd.DataFrame(data)
    
    #Assuming the features are passed in the correct order
    features = df[['age', 'total_visits', 'average_purchase_value', 'last_purchase_days_ago']]
    
    #Make predictions
    predictions = predict(model, features)
    
    #Convert predictions to a List
    predictions_list = predictions.tolist()
    
    return jsonify(predictions=predictions_list)
    
if __name__ == '__main__':
    app.run(debug=True)