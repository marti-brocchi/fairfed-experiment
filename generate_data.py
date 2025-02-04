import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from aif360.sklearn.datasets import fetch_adult


def get_numerical_dataset():
    """
    Import the Adult Income Dataset
    """
    X, y, sample_weights = fetch_adult(dropna=False)

    # # Shuffle the dataset
    # X, y, sample_weights = shuffle(X, y, sample_weights, random_state=42)

    # Transform sex into binary value
    # 0 if Female, 1 if Male
    X["sex"] = (X["sex"] == "Male").astype(int)

    # Encode other features
    le = LabelEncoder()
    for column in X.columns:
        if X[column].dtype != np.float64 and X[column].dtype != np.int64:
            X[column] = le.fit_transform(X[column])

    y = le.fit_transform(y)

    return X, y


def manage_protected_attributes(X, y, sensitive_attributes):
    """
    Saves and removes the protected attributes from the dataset

    Input:
    - `X`: features
    - `y`: labels
    - `sensitive_attributes`: list of sensitive attributes to remove from the dataset

    Output:
    - `X`: features without the protected attributes
    - `y`: labels
    - `protected_attribute_values`: the values of the protected attributes
    - `privileged`: the set of privileged groups
    """

    protected_attribute_values = X["sex"].values
    groups = set(protected_attribute_values)
    privileged_groups = {1}

    # remove protected attributes from features
    for sensitive_attribute in sensitive_attributes:
        if sensitive_attribute in X.columns:
            X = X.drop(columns=[sensitive_attribute])

    return X, y, protected_attribute_values, privileged_groups


def split_dataset(X, y, test_size=0.3):
    """
    Split the dataset into training and testing sets

    Input:
    - `X`: features
    - `y`: labels
    - `test_size`: the size of the testing set, expressed as a percentage of the dataset (default: 0.2)

    Output:
    - `X_train`: training features
    - `y_train`: training labels
    - `X_test`: testing features
    - `y_test`: testing labels
    """

    features = X.values
    labels = y

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Convert features and labels to correct data types
    X_scaled = X_scaled.astype(np.float32)
    y = labels.astype(np.int32)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, shuffle=False
    )

    return X_train, y_train, X_test, y_test


def save_clients_data(num_clients, client_train_data, client_test_data, data_dir):
    for client in range(num_clients):
        # Create directory for each client
        client_dir = f"client_{client}"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(f"{data_dir}/{client_dir}"):
            os.makedirs(f"{data_dir}/{client_dir}")

        # Save the data
        features, labels, _, _ = client_train_data[client]
        np.save(f"{data_dir}/{client_dir}/train_data.npy", features)
        np.save(f"{data_dir}/{client_dir}/train_target.npy", labels)

        features, labels, _, _ = client_test_data[client]
        np.save(f"{data_dir}/{client_dir}/valid_data.npy", features)
        np.save(f"{data_dir}/{client_dir}/valid_target.npy", labels)


####################################################
#                      SETUPS                      #
####################################################
def no_cov_same_size(features, labels, protected_attribute_values, num_clients: int):
    client_data = []
    client_size = features.shape[0] // num_clients

    for i in range(num_clients):
        client_features = features[i * client_size : (i + 1) * client_size]
        client_labels = labels[i * client_size : (i + 1) * client_size]
        client_protected_values = protected_attribute_values[
            i * client_size : (i + 1) * client_size
        ]

        # Count the number of examples for each class and associate it as metadata
        client_metadata = {
            f"{label}": count
            for label, count in enumerate(np.bincount(client_protected_values))
        }
        client_data.append(
            (client_features, client_labels, client_protected_values, client_metadata)
        )

    return client_data


####################################################
#                       MAIN                       #
####################################################


def main():
    num_clients = 5
    data_dir = "adult-data"

    X, y = get_numerical_dataset()
    X, y, protected_attribute_values, _ = manage_protected_attributes(
        X, y, ["sex", "race"]
    )
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    client_train_data = no_cov_same_size(
        X_train, y_train, protected_attribute_values, num_clients
    )

    client_test_data = no_cov_same_size(
        X_test, y_test, protected_attribute_values, num_clients
    )

    save_clients_data(num_clients, client_train_data, client_test_data, data_dir)


if __name__ == "__main__":
    main()
