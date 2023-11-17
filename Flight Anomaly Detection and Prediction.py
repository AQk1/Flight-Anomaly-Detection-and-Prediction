from faker import Faker
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random

fake = Faker()

def generate_fake_flight_data(num_records=1000):
    flight_data = {
        'flight_number': [fake.random_int(min=100, max=9999) for _ in range(num_records)],
        'departure_city': [fake.city() for _ in range(num_records)],
        'arrival_city': [fake.city() for _ in range(num_records)],
        'departure_time': [fake.date_time_this_decade() for _ in range(num_records)],
        'arrival_time': [fake.date_time_this_decade() for _ in range(num_records)],
        'distance': [random.uniform(100, 5000) for _ in range(num_records)],  # Example distance in miles
        'passenger_count': [fake.random_int(min=50, max=300) for _ in range(num_records)],
    }

    return pd.DataFrame(flight_data)

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
    dataset['data_squared'] = dataset['distance'] ** 2
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
    model.fit(train_data[['distance']])
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
    test_predictions = model.predict(test_data[['distance']])
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
    plt.scatter(test_data['distance'], test_labels, c=np.where(model.predict(test_data[['distance']]) == -1, 'red', 'green'), label='Predictions')
    plt.xlabel('Flight Distance')
    plt.ylabel('Label (0: Normal, 1: Anomalous)')
    plt.legend()
    plt.title('Flight Anomaly Detection')
    plt.show()

def main():
    # Step 1: Generate fake flight data
    fake_flight_data = generate_fake_flight_data(num_records=2000)

    # Save the generated data to a CSV file
    fake_flight_data.to_csv('fake_flight_data.csv', index=False)

    # Step 2: Load fake flight data
    file_path = "fake_flight_data.csv"
    flight_data = load_flight_data(file_path)

    # Step 3: Feature Engineering
    flight_data = perform_feature_engineering(flight_data)

    # Step 4: Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(flight_data[['distance']], flight_data['passenger_count'], test_size=0.2, random_state=42)

    # Step 5: Train the Isolation Forest model
    isolation_forest_model = train_isolation_forest(train_data)

    # Step 6: Evaluate the model
    evaluate_model(isolation_forest_model, test_data, test_labels)

    # Step 7: Visualize the results
    visualize_results(test_data, test_labels, isolation_forest_model)

if __name__ == "__main__":
    main()
