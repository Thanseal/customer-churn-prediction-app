import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Load the processed data
print("Loading processed data...")
data = pd.read_csv('data/processed_data.csv')

# Assume 'features' and 'label' are columns in your processed_data.csv
X = data[['age', 'total_visits', 'average_purchase_value', 'last_purchase_days_ago']]  # Update these feature names
y = data['churn']  # Update with the correct label column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1_score = f1_score(y_test, predictions)

print("-------------------------------------------------------------------")

print("Training completed. Model performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

print("-------------------------------------------------------------------")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest:\n", classification_report(y_test, rf_preds))

print("-------------------------------------------------------------------")

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)
print("Gradient Boosting:\n", classification_report(y_test, gb_preds))

print("-------------------------------------------------------------------")

#param_grid = {
#    'n_estimators': [100, 200, 300],
 #   'max_depth': [3, 5, 7],
  #  'min_samples_split': [2, 5, 10]
#}

#grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

#print("Best Parameters found : ", grid_search.best_params_) 

print("-------------------------------------------------------------------")

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X);

from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(rf_model)
X_selected = selector.fit_transform(X, y)

X_selected_train, X_selected_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_selected_train, y_train)

# Predict and evaluate
predictions = model.predict(X_selected_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
#f1_score = f1_score(y_test, predictions)
print("Training completed. Model performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("-------------------------------------------------------------------")
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
#shap.summary_plot(shap_values, X_test)

#from lime.lime_tabular import LimeTabularExplainer

# Example feature names - replace with your actual feature names
#feature_names = ['age', 'total_visits', 'average_purchase_value', 'last_purchase_days_ago']

# Initialize LIME explainer
#explainer = LimeTabularExplainer(
#    training_data=X_train, 
#    feature_names=feature_names, 
#    class_names=['No Churn', 'Churn'], 
#    mode='classification'
#)

# Example explanation
#instance = X_test[0]  # Example instance from the test set
#explanation = explainer.explain_instance(instance, model.predict_proba)
#explanation.show_in_notebook()

# Save the model
with open('model.pk1', 'wb') as file:
    pickle.dump(rf_model, file)

