from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred):
    """Evaluate the model and return key metrics."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}

def plot_feature_importance(model, feature_names):
    """Plot the feature importance for the model."""
    importance = model.coef_[0]
    plt.barh(feature_names, importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

def save_metrics(metrics, file_path):
    """Save evaluation metrics to a JSON file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(metrics, f)
