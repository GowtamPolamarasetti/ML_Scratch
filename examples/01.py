# my_ml_project/examples/11_neural_network_mnist.py
# The "Final Exam": Train our NN on the MNIST dataset!

print("Loading dependencies...")
import numpy as np

try:
    from sklearn.datasets import fetch_openml
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Use our own library! ---
from pyml.model_selection import train_test_split 
from pyml.neural_network import Sequential, Dense, ActivationLayer, Flatten
from pyml.neural_network._activations import ReLU
from pyml.neural_network._losses import SoftmaxCrossEntropyLoss
from pyml.neural_network._optimizers import Adam
from pyml.metrics import accuracy_score
# --- End pyml imports ---

def load_and_prepare_data():
    """Loads and preprocesses the MNIST dataset."""
    print("Fetching MNIST data... (This may take a minute)")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    
    print("Data fetched. Preprocessing...")
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # We'll use a smaller subset to make training faster
    # 1. Create a 12,000 sample subset
    np.random.seed(42) # for reproducibility
    n_total = X.shape[0]
    subset_indices = np.random.choice(n_total, 12000, replace=False)
    X_subset = X[subset_indices]
    y_subset = y[subset_indices]
    
    # 2. Use our own train_test_split to get 10k train, 2k test
    # test_size = 2000 / 12000 = 1/6
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=(1/6), random_state=42
    )
    
    print(f"Training on {X_train.shape[0]} samples.")
    print(f"Testing on {X_test.shape[0]} samples.")
    
    return X_train, X_test, y_train, y_test

def build_model():
    """Builds our Sequential model."""
    print("Building model...")
    model = Sequential()
    
    # Input layer: 784 features (28*28)
    model.add(Dense(n_input=784, n_output=128, random_state=42))
    model.add(ActivationLayer(ReLU())) # <-- Use correct ActivationLayer
    
    # Hidden layer
    model.add(Dense(n_input=128, n_output=64, random_state=42))
    model.add(ActivationLayer(ReLU())) # <-- Use correct ActivationLayer
    
    # Output layer: 10 classes (digits 0-9)
    # We output raw logits, as SoftmaxCrossEntropyLoss handles activation
    model.add(Dense(n_input=64, n_output=10, random_state=42))
    
    print("Model built successfully.")
    return model

def main():
    if not SKLEARN_AVAILABLE:
        print("This example requires scikit-learn to fetch MNIST.")
        print("Please run: pip install scikit-learn")
        return

    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # 2. Build Model
    model = build_model()
    
    # 3. Compile Model
    print("Compiling model...")
    model.compile(
        loss=SoftmaxCrossEntropyLoss(),
        optimizer_class=Adam,
        learning_rate=0.001
    )
    
    # 4. Train Model
    print("\n--- Starting Training ---")
    model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=64,
        validation_data=(X_test, y_test)
    )
    print("--- Training Complete ---")

    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Final Results ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Check for a good result
    if accuracy > 0.9:
        print("SUCCESS: Model achieved > 90% accuracy!")
    else:
        print("Result: Model trained. Accuracy is < 90%.")

if __name__ == "__main__":
    main()