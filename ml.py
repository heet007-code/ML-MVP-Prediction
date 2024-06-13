import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
mvps = pd.read_csv("mvps.csv")
teams = pd.read_csv("teams.csv")
player_mvp_stats = pd.read_csv("player_mvp_stats.csv")

# Display the first few rows of each DataFrame to understand their structure
print(mvps.head())
print(teams.head())
print(player_mvp_stats.head())

# Example of merging data
# Assuming 'player_id' is a common key to merge player_mvp_stats and mvps
data = pd.merge(player_mvp_stats, mvps, on='player_id', how='left')
data = pd.merge(data, teams, on='team_id', how='left')

# Preprocess data
# Fill missing values, encode categorical features, etc.
data.fillna(0, inplace=True)  # Example of filling missing values

# Encoding categorical features
label_encoders = {}
categorical_columns = ['team_name', 'player_name']  # Example categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target
features = data.drop(columns=['mvp'])  # Drop the target column
target = data['mvp']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(report)

import joblib

# Save the trained model to a file
joblib.dump(model, 'mvp_prediction_model.joblib')

# Load the model from the file
model = joblib.load('mvp_prediction_model.joblib')

# Make predictions on new data (ensure new data is preprocessed similarly)
new_data = pd.DataFrame({  # Replace with actual new data
    'player_id': [1, 2],
    'team_id': [1, 2],
    # Add other feature columns as needed
})
# Preprocess new data similarly to training data
new_data.fillna(0, inplace=True)
for col in categorical_columns:
    new_data[col] = label_encoders[col].transform(new_data[col])

# Make predictions
predictions = model.predict(new_data)
print(predictions)
