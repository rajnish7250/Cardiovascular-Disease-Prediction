from src.preprocess import load_data, preprocess_data
from src.train import train_models, save_model
from src.evaluate import evaluate_models

# Load data
df = load_data("data/heart.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train models
models = train_models(X_train, y_train)

# Evaluate
best_model_name = evaluate_models(models, X_test, y_test)

# Save best model
best_model = models[best_model_name]
save_model(best_model)

print("\nModel saved successfully!")