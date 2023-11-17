import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_flight_data(file_path):
    """
    Load flight data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded flight data.
    """
    flight_data = pd.read_csv(file_path)
    return flight_data

def perform_feature_engineering(dataset):
    """
    Perform feature engineering on the dataset.

    Parameters:
    - dataset (pd.DataFrame): Flight data.

    Returns:
    - pd.DataFrame: Feature-engineered dataset.
    """
    # Simulated feature engineering process (replace with actual feature engineering logic)
    dataset['data_squared'] = dataset['data'] ** 2
    return dataset

def train_isolation_forest(train_data):
    """
    Train an Isolation Forest model.

    Parameters:
    - train_data (pd.DataFrame): Training data.

    Returns:
    - IsolationForest: Trained Isolation Forest model.
    """
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(train_data[['data']])
    return model

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the Isolation Forest model.

    Parameters:
    - model (IsolationForest): Trained model.
    - test_data (pd.DataFrame): Test data.
    - test_labels (pd.Series): True labels for the test data.

    Returns:
    - None
    """
    test_predictions = model.predict(test_data[['data']])
    print("Classification Report:")
    print(classification_report(test_labels, np.where(test_predictions == -1, 1, 0)))

def visualize_results(test_data, test_labels, model):
    """
    Visualize the results of the Isolation Forest model.

    Parameters:
    - test_data (pd.DataFrame): Test data.
    - test_labels (pd.Series): True labels for the test data.
    - model (IsolationForest): Trained model.

    Returns:
    - None
    """
    plt.scatter(test_data['data'], test_labels, c=np.where(model.predict(test_data[['data']]) == -1, 'red', 'green'), label='Predictions')
    plt.xlabel('Flight Data')
    plt.ylabel('Label (0: Normal, 1: Anomalous)')
    plt.legend()
    plt.title('Flight Anomaly Detection')
    plt.show()

def main():
    # Step 1: Load real flight data
    file_path = "path/to/your/flight/data.csv"  # Replace with the actual path to your CSV file
    flight_data = load_flight_data(file_path)

    # Step 2: Optional Feature Engineering
    flight_data = perform_feature_engineering(flight_data)

    # Step 3: Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(flight_data[['data']], flight_data['label'], test_size=0.2, random_state=42)

    # Step 4: Train the Isolation Forest model
    isolation_forest_model = train_isolation_forest(train_data)

    # Step 5: Evaluate the model
    evaluate_model(isolation_forest_model, test_data, test_labels)

    # Step 6: Visualize the results
    visualize_results(test_data, test_labels, isolation_forest_model)

if __name__ == "__main__":
    main()
