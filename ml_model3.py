# Import necessary libraries
import pandas as pd                             # For data manipulation
import numpy as np                              # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # For encoding and normalization
from sklearn.impute import SimpleImputer        # For handling missing values
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # Ensemble models
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, log_loss

# Optional: Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer

import pandas as pd

# File path to your dataset
file_path = 'D:\Project expo\Insurance sample data\A1 Data.xlsx'

# Read the Excel file and store it in the variable 'data'
data = pd.read_excel(file_path)

# Display the first few rows to confirm
print("Dataset Loaded Successfully!")
print(data.head())

# Step 2: Replace Missing Values

# Separate numerical and categorical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Imputer for numerical columns (replace missing with mean)
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Imputer for categorical columns (replace missing with most frequent value)
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Display updated dataset
print("Missing values replaced successfully!")
print(data.isnull().sum())  # Confirm no missing values remain

from sklearn.preprocessing import LabelEncoder

# Step 3: Convert Nominal (Categorical) to Numerical
label_encoders = {}  # Dictionary to store encoders for each column

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the label encoder for future reference

# Display the updated dataset
print("Nominal to Numerical Conversion Complete!")
print(data.head())

from sklearn.preprocessing import MinMaxScaler

# Step 4: Normalize Selected Attributes
scaler = MinMaxScaler()

# List of selected attributes to normalize (exclude 'Claim Number' and 'ClaimantAge')
selected_columns = ["Vehicle Flag (1=Motor Vehicle Involved)"]

# Apply MinMaxScaler to the selected columns
data[selected_columns] = scaler.fit_transform(data[selected_columns])

# Display updated dataset
print("Normalization Complete!")
print(data.head())

from sklearn.model_selection import train_test_split

# Step 5: Split Data into Training and Testing Sets

# Define the target column and features
target_column = "Fraud Flag (1=Yes 0=No)"
X = data.drop(columns=[target_column])  # Features
y = data[target_column].astype(int)  # Target variable

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Display the shape of the split data
print("Data Splitting Complete!")
print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, Testing Labels Shape: {y_test.shape}")

from sklearn.linear_model import LogisticRegression

# Step 6: Classification by Regression

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

# Train the model using the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the model's performance
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_logistic))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Assuming y_test and y_pred_logistic are your true and predicted labels for fraud detection
conf_matrix = confusion_matrix(y_test, y_pred_logistic)

# Print classification report
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_logistic))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.title('Confusion Matrix for Fraud Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Step 7: Decision Tree Classification

# Initialize the Decision Tree model with specified parameters
decision_tree_model = DecisionTreeClassifier(
    criterion='entropy',  # 'gain_ratio' is not directly available, use 'entropy' as an alternative
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2
)

# Step 8: Replace Missing Values in Target Column

# Initialize the imputer for the target column
target_imputer = SimpleImputer(strategy='mean')

# Apply the imputer to the target column
y_train = target_imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Ensure the imputed values are integers
y_train = y_train.round().astype(int)

# Now you can proceed with training models using y_train
# For example, training the Decision Tree model
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate the model's performance
print("Decision Tree Performance:")
print(classification_report(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))

from sklearn.ensemble import RandomForestClassifier

# Step 9: Random Forest Classification

# Initialize the Random Forest model with specified parameters
random_forest_model = RandomForestClassifier(
    n_estimators=10,
    criterion='entropy',  # 'gain_ratio' is not directly available, use 'entropy' as an alternative
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2
)

# Train the model using the training data
random_forest_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_forest = random_forest_model.predict(X_test)

# Evaluate the model's performance
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_forest))
print(confusion_matrix(y_test, y_pred_forest))

# Step 10: Apply Model to New Data

# Assuming you have a new dataset to apply the model on
# For demonstration, let's use the test set as the new data
new_data = X_test  # Replace with your actual new dataset

# Apply the logistic regression model to the new data
new_predictions = logistic_model.predict(new_data)

# Display the predictions
print("Predictions from Logistic Regression on New Data:")
print(new_predictions)

# Step 11: Apply Decision Tree Model to New Data

# Apply the decision tree model to the new data
new_predictions_tree = decision_tree_model.predict(new_data)

# Display the predictions
print("Predictions from Decision Tree on New Data:")
print(new_predictions_tree)

# Step 12: Apply Random Forest Model to New Data

# Apply the random forest model to the new data
new_predictions_forest = random_forest_model.predict(new_data)

# Display the predictions
print("Predictions from Random Forest on New Data:")
print(new_predictions_forest)

from sklearn.ensemble import VotingClassifier

# Step 13: Group Models Using Voting Classifier

# Initialize the Voting Classifier with the three models
voting_model = VotingClassifier(
    estimators=[
        ('logistic', logistic_model),
        ('decision_tree', decision_tree_model),
        ('random_forest', random_forest_model)
    ],
    voting='soft'  # Use 'soft' voting to enable predict_proba
)

# Train the voting model using the training data
voting_model.fit(X_train, y_train)

# Apply the voting model to the new data
new_predictions_voting = voting_model.predict(new_data)

# Display the predictions
print("Predictions from Voting Classifier on New Data:")
print(new_predictions_voting)   

# Step 14: Performance Evaluation for Logistic Regression

# Calculate performance metrics
accuracy = accuracy_score(y_test, new_predictions)
precision, recall, _, _ = precision_recall_fscore_support(y_test, new_predictions, average='weighted')
mae = mean_absolute_error(y_test, new_predictions)
mse = mean_squared_error(y_test, new_predictions)
rmse = np.sqrt(mse)  # Manually calculate RMSE
cross_entropy = log_loss(y_test, logistic_model.predict_proba(new_data))

# Display the performance metrics
print("Performance Metrics for Logistic Regression:")
print(f"Accuracy: {accuracy}")
print(f"Weighted Mean Recall: {recall}")
print(f"Weighted Mean Precision: {precision}")
print(f"Normalized Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Cross-Entropy: {cross_entropy}")

# Step 15: Performance Evaluation for Decision Tree

# Calculate performance metrics
accuracy_tree = accuracy_score(y_test, new_predictions_tree)
precision_tree, recall_tree, _, _ = precision_recall_fscore_support(y_test, new_predictions_tree, average='weighted')
mae_tree = mean_absolute_error(y_test, new_predictions_tree)
mse_tree = mean_squared_error(y_test, new_predictions_tree)
rmse_tree = np.sqrt(mse_tree)  # Manually calculate RMSE
cross_entropy_tree = log_loss(y_test, decision_tree_model.predict_proba(new_data))

# Display the performance metrics
print("Performance Metrics for Decision Tree:")
print(f"Accuracy: {accuracy_tree}")
print(f"Weighted Mean Recall: {recall_tree}")
print(f"Weighted Mean Precision: {precision_tree}")
print(f"Normalized Absolute Error: {mae_tree}")
print(f"Root Mean Squared Error: {rmse_tree}")
print(f"Cross-Entropy: {cross_entropy_tree}")

# Step 16: Performance Evaluation for Random Forest

# Calculate performance metrics
accuracy_forest = accuracy_score(y_test, new_predictions_forest)
precision_forest, recall_forest, _, _ = precision_recall_fscore_support(y_test, new_predictions_forest, average='weighted')
mae_forest = mean_absolute_error(y_test, new_predictions_forest)
mse_forest = mean_squared_error(y_test, new_predictions_forest)
rmse_forest = np.sqrt(mse_forest)  # Manually calculate RMSE
cross_entropy_forest = log_loss(y_test, random_forest_model.predict_proba(new_data))

# Display the performance metrics
print("Performance Metrics for Random Forest:")
print(f"Accuracy: {accuracy_forest}")
print(f"Weighted Mean Recall: {recall_forest}")
print(f"Weighted Mean Precision: {precision_forest}")
print(f"Normalized Absolute Error: {mae_forest}")
print(f"Root Mean Squared Error: {rmse_forest}")
print(f"Cross-Entropy: {cross_entropy_forest}")

# Step 17: Performance Evaluation for Voting Classifier

# Calculate performance metrics
accuracy_voting = accuracy_score(y_test, new_predictions_voting)
precision_voting, recall_voting, _, _ = precision_recall_fscore_support(y_test, new_predictions_voting, average='weighted')
mae_voting = mean_absolute_error(y_test, new_predictions_voting)
mse_voting = mean_squared_error(y_test, new_predictions_voting)
rmse_voting = np.sqrt(mse_voting)  # Manually calculate RMSE
cross_entropy_voting = log_loss(y_test, voting_model.predict_proba(new_data))

# Display the performance metrics
print("Performance Metrics for Voting Classifier:")
print(f"Accuracy: {accuracy_voting}")
print(f"Weighted Mean Recall: {recall_voting}")
print(f"Weighted Mean Precision: {precision_voting}")
print(f"Normalized Absolute Error: {mae_voting}")
print(f"Root Mean Squared Error: {rmse_voting}")
print(f"Cross-Entropy: {cross_entropy_voting}")

import matplotlib.pyplot as plt

# Group the data by 'Body Part' and sum the 'Fraud Flag (1=Yes 0=No)'
fraud_by_body_part = data.groupby('Body Part')['Fraud Flag (1=Yes 0=No)'].sum()

# Define the body part names
body_part_names = [
    "Finger", "Hand", "Knee", "Wrist", "Multiple", "Back", "Arm", "Neck", "Leg", "Foot",
    "Spine", "Ankle", "Groin", "Head", "Shoulder", "Eye", "Abdomen", "Face", "Torso",
    "Hip", "Jaw", "Toes", "Tooth", "Elbow", "Ribs", "Unknown"
]

# Plot the data
plt.figure(figsize=(12, 6))
fraud_by_body_part.plot(kind='bar', color='skyblue')
plt.title('Number of Fraud Flags by Body Part')
plt.xlabel('Body Part')
plt.ylabel('Number of Fraud Flags')
plt.xticks(ticks=range(len(body_part_names)), labels=body_part_names, rotation=45, ha='right')
plt.tight_layout()
plt.show()

import joblib

# Save the logistic regression model
joblib.dump(logistic_model, 'logistic_model.pkl')

# Save the decision tree model
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')

# Save the random forest model
joblib.dump(random_forest_model, 'random_forest_model.pkl')

# Save the voting classifier model
joblib.dump(voting_model, 'voting_model.pkl')

# Load the logistic regression model
loaded_logistic_model = joblib.load('logistic_model.pkl')

# Load the decision tree model
loaded_decision_tree_model = joblib.load('decision_tree_model.pkl')

# Load the random forest model
loaded_random_forest_model = joblib.load('random_forest_model.pkl')

# Load the voting classifier model
loaded_voting_model = joblib.load('voting_model.pkl')

# Define the hierarchical mapping for 'Body Part'
hierarchy_mapping = {
    "Head": 1,
    "Neck": 2,
    "Face": 3,
    "Spine": 4,
    "Multiple": 5,
    "Back": 6,
    "Ribs": 7,
    "Torso": 8,
    "Hip": 9,
    "Shoulder": 10,
    "Hands": 11,
    "Knee": 12,
    "Wrists": 13,
    "Finger": 14,
    "Toes": 15
}

# Create a new column 'Body_Part_Hierarchy' based on the mapping
data['Body_Part_Hierarchy'] = data['Body Part'].map(hierarchy_mapping)

# Display the first few rows to confirm the mapping
print("Hierarchical classification applied to 'Body Part':")
print(data[['Body Part', 'Body_Part_Hierarchy']].head())

# Now use the 'Body_Part_Hierarchy' column as part of the features
# You can include it as an additional feature in X
X = data.drop(columns=[target_column])  # Features
X['Body_Part_Hierarchy'] = data['Body_Part_Hierarchy']

# Step 1: Ensure the hierarchical column is present (this should be done previously)
# If not already done, map 'Body Part' to the hierarchy values
data['Body_Part_Hierarchy'] = data['Body Part'].map(hierarchy_mapping)

# Step 2: Sort the entire dataset by the 'Body_Part_Hierarchy' column in ascending order
sorted_data = data.sort_values(by='Body_Part_Hierarchy', ascending=True)

# Display the first few rows of the sorted data to confirm
print("Dataset sorted by Body Part hierarchy:")
print(sorted_data[['Body Part', 'Body_Part_Hierarchy']].head())

# Optionally, save the sorted dataset to a new Excel file for further analysis or reporting
# sorted_file_path = '"D:\Project expo\Insurance sample data\Project_Exhibition storing excel data\Store_1.xlsx"'
# sorted_data.to_excel(sorted_file_path, index=False)
# print(f"Sorted dataset saved to: {sorted_file_path}")

# Step 1: Ensure the hierarchical column for 'Body Part' is present
data['Body_Part_Hierarchy'] = data['Body Part'].map(hierarchy_mapping)

# Step 2: Map 'Witness Present' to numeric values (Yes = 1, No = 0)
data['Witness_Present_Flag'] = data['WitnessPresent'].map({'Yes': 1, 'No': 0})

# Step 3: Sort the dataset first by 'Witness_Present_Flag' (descending), then by 'Body_Part_Hierarchy' (ascending)
# This ensures rows with a witness present are ranked higher, and within those, body part hierarchy is respected.
sorted_data = data.sort_values(by=['Witness_Present_Flag', 'Body_Part_Hierarchy'], ascending=[False, True])

# Display the first few rows of the sorted data to confirm
print("Dataset sorted by Witness Present and Body Part hierarchy:")
print(sorted_data[['Body Part', 'WitnessPresent', 'Body_Part_Hierarchy', 'Witness_Present_Flag']].head())

# Define the mapping for the "ClaimantMaritalStatus" column
status_mapping = {
    0: '',
    1: 'Single',
    2: 'Married',
    3: 'Divorced'
}
data['ClaimantMaritalStatus'] = data['ClaimantMaritalStatus'].map(status_mapping)


# Proceed with saving the sorted dataset to a new Excel file
sorted_file_path = 'D:\Project expo\Insurance sample data\Project_Exhibition storing excel data\Store_1.xlsx'
sorted_data.to_excel(sorted_file_path, index=False)
print(f"Sorted dataset saved to: {sorted_file_path}")
import random
claim_numbers = list(range(4001, 8008))
random.shuffle(claim_numbers)
data['Claim Number'] = claim_numbers[:len(data)]
# Assuming 'MonthClaimed' is a column in your DataFrame
# Get unique months from the 'MonthClaimed' column
unique_months = data['MonthClaimed'].unique()

random.shuffle(unique_months)

month_mapping = {original: shuffled for original, shuffled in zip(data['MonthClaimed'].unique(), unique_months)}

data['MonthClaimed'] = data['MonthClaimed'].map(month_mapping)
# Define the mapping for months
month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

# Map the numbers in 'MonthClaimed' to month names
data['MonthClaimed'] = data['MonthClaimed'].map(month_mapping)

# Display the first few rows to confirm the mapping
print("MonthClaimed column converted to month names:")
print(data[['MonthClaimed']].head())

# Save the sorted dataset to a new Excel file
sorted_file_path = 'D:\Project expo\Insurance sample data\Project_Exhibition storing excel data\Store_1.xlsx'
sorted_data.to_excel(sorted_file_path, index=False)
print(f"Sorted dataset saved to: {sorted_file_path}")

# Define priority criteria
def calculate_priority(row):
    # Assign weights to ClaimantMaritalStatus
    marital_status_priority = {
        3: 100,  # Highest precedence
        2: 10,
        1: 1,
        0: 0   # Lowest precedence
    }
    
    # Assign weights to Nature of Injury
    injury_priority = {
        "Death": 1000,
        "Puncture": 900,
        "Infection/Disease": 800,
        "Mental Health": 700,
        "Animal/Insect Bite": 600,
        "Burn": 500,
        "Abrasion": 400,
        "Laceration": 300,
        "Fracture": 200,
        "Repetitive Motion": 100,
        "Contusion": 50,
        "Sprain/Strain": 10
    }
    
    # Calculate priority based on criteria
    witness_priority = 1 if row.get('WitnessPresent', 'No') == 'Yes' else 0
    body_part_priority = row.get('Body_Part_Hierarchy', 0)
    marital_priority = marital_status_priority.get(row.get('ClaimantMaritalStatus', 0), 0)
    vehicle_priority = 50 if row.get('Vehicle Flag (1=Motor Vehicle Involved)', 0) == 1 else 0
    injury_priority_value = injury_priority.get(row.get('Nature of Injury', ''), 0)
    
    # Combine priorities
    return marital_priority + witness_priority * 10 + body_part_priority + vehicle_priority + injury_priority_value

# Add a priority column
data['Priority'] = data.apply(calculate_priority, axis=1)

# Sort by priority
sorted_data = data.sort_values(by='Priority', ascending=False)

# Display the first few rows of the sorted data to confirm
print("Dataset sorted by Priority:")
print(sorted_data[['Body Part', 'WitnessPresent', 'Body_Part_Hierarchy', 'ClaimantMaritalStatus', 'Vehicle Flag (1=Motor Vehicle Involved)', 'Nature of Injury', 'Priority']].head())

# Save the sorted dataset to a new Excel file
sorted_file_path = 'D:\\Project expo\\Insurance sample data\\Project_Exhibition storing excel data\\Store_1.xlsx'
sorted_data.to_excel(sorted_file_path, index=False)
print(f"Sorted dataset saved to: {sorted_file_path}")


# Define the specific Nature of Injury statements and their corresponding integer codes
injury_statements = [
    "Death", "Puncture", "Infection/Disease", "Mental Health", "Animal/Insect Bite",
    "Burn", "Abrasion", "Laceration", "Fracture", "Repetitive Motion",
    "Contusion", "Sprain/Strain"
]

# Assign a unique integer to each injury statement
injury_mapping = {injury: i+1 for i, injury in enumerate(injury_statements)}

# Map the 'Nature of Injury' column to the defined integers
data['Injury_Code'] = data['Nature of Injury'].map(injury_mapping)

# Display the mapping
print("Injury to Integer Mapping:")
for injury, code in injury_mapping.items():
    print(f"{injury}: {code}")

# Display the first few rows to confirm the mapping
print(data[['Nature of Injury', 'Injury_Code']].head())

# Define the specific Nature of Injury statements and their corresponding integer codes
injury_statements = [
    "Death", "Puncture", "Infection/Disease", "Mental Health", "Animal/Insect Bite",
    "Burn", "Abrasion", "Laceration", "Fracture", "Repetitive Motion",
    "Contusion", "Sprain/Strain"
]

# Create a mapping from integers to injury statements
reverse_injury_mapping = {i+1: injury for i, injury in enumerate(injury_statements)}

# Map the 'Injury_Code' column back to the injury statements
data['Nature of Injury'] = data['Injury_Code'].map(reverse_injury_mapping)

# Display the first few rows to confirm the mapping
print("Nature of Injury column converted back to string values:")
print(data[['Injury_Code', 'Nature of Injury']].head())

# Save the sorted dataset to a new Excel file
sorted_file_path = 'D:\\Project expo\\Insurance sample data\\Project_Exhibition storing excel data\\Store_1.xlsx'
data.to_excel(sorted_file_path, index=False)
print(f"Sorted dataset saved to: {sorted_file_path}")




