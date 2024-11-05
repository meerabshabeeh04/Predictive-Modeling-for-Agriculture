import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
crops = pd.read_csv("soil_measures.csv")
# Check for missing values in the dataset and print the count per column
print(crops.isna().sum())
# Print unique crop types in the dataset
print(crops['crop'].unique())
# Separate the dataset into features (excluding the target 'crop' column) and target
features = crops.drop(columns=['crop'])  # All columns except 'crop' are used as features
target = crops['crop']  # 'crop' column is used as the target variable
# Split the data into training and testing sets, with 50% of the data in each set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=41)
# Initialize a dictionary to store the F1-score performance for each feature
features_dict = {}
# Loop through a list of selected features ("N", "P", "K", "ph") to evaluate their performance individually
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression()  # Initialize Logistic Regression model
    log_reg.fit(X_train[[feature]], y_train)  # Train the model on the selected feature
    # Predict the target for the test set based on the single feature
    y_pred = log_reg.predict(X_test[[feature]])
    # Calculate the weighted F1-score to evaluate the feature's performance
    feature_performance = f1_score(y_test, y_pred, average="weighted")
    features_dict[feature] = feature_performance  # Store the F1-score in the dictionary
    # Print the F1-score for the current feature
    print(f"F1-score for {feature}: {feature_performance}")
# Determine the feature with the highest F1-score
max_score = max(features_dict.values())  # Find the highest score among features
best_predictive_feature = {}
# Loop through features_dict to find features with F1-score equal to the maximum score
for feature in features_dict:
    if features_dict[feature] == max_score:
        best_predictive_feature[feature] = max_score  # Add to best features dictionary if score is max
# Print the best predictive feature(s) and their corresponding F1-score
print(best_predictive_feature)
