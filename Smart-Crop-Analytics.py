# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('crop_yield.csv')

# Preprocess the dataset (one-hot encoding for categorical variables)
data_encoded = pd.get_dummies(data)

# Separate features and target variable
features = data_encoded.drop(columns=['Yield'])
target = data_encoded['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}\n")

    plt.figure()
    plt.scatter(y_test, predictions, alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title(f'{model.__class__.__name__}: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# Evaluate Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
evaluate_model(gb_model, X_train, X_test, y_train, y_test)

# Evaluate Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
evaluate_model(dt_model, X_train, X_test, y_train, y_test)

# Preprocessing for SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).drop(columns=['Yield']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
])

svr_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('svr', SVR(kernel='rbf'))
])

# Evaluate Support Vector Regressor
evaluate_model(svr_pipeline, X_train, X_test, y_train, y_test)

print("Models evaluated successfully.")
