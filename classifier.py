import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_and_preprocess_data(filepath="/Users/krishna/Desktop/Iris Flower Classification/data/Iris.csv"):
    """
    Loads the Iris dataset, encodes the target labels, and scales the features.
    """
    # 1. Load dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {filepath}.")
        print("Please double-check the path and make sure 'Iris.csv' is inside the 'data' folder.")
        return None, None, None, None, None
    
    # 2. Drop the 'Id' column as it's not a predictive feature
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    # 3. Separate features (X) and target (y)
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # 4. Encode target labels (e.g., 'Iris-setosa' -> 0)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 5. Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 6. Standardize features for optimal neural network performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_

def main():
    # Load and prepare data
    print("Loading and preprocessing data...")
    # The function now defaults to your specific absolute path
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess_data() 
    
    if X_train is None:
        return # Exit if data failed to load
    
    # Model 1: Standard Machine Learning (Random Forest)
    print("\n--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_predictions = rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, rf_predictions, target_names=target_names))

    # Model 2: Deep Learning (Multi-Layer Perceptron)
    print("\n--- Training MLP Classifier ---")
    # Architecture: 2 hidden layers with 16 nodes each, using ReLU activation
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(16, 16), 
        activation='relu', 
        solver='adam', 
        max_iter=1000, 
        random_state=42
    )
    mlp_model.fit(X_train, y_train)
    
    mlp_predictions = mlp_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, mlp_predictions):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, mlp_predictions, target_names=target_names))

if __name__ == "__main__":
    main()
