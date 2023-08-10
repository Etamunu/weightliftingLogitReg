import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######Load and format the data##################################
data = pd.read_csv('E:/science/enseignement/expose 2/weightlifting/weightlifting_dataset.csv') # put the location of the dataset on your computer
data['hips'] = data['hips'].replace({'yes': 1, 'no': 0}) # correct the data format
data['sex'] = data['sex'].replace({'f': 0, 'm': 1}) # correct the data format
data['injury'] = data[['shoulder', 'knees', 'back', 'wrist', 'hips']].max(axis=1) # regroup all types of injury
data['train_total'] = data['train_days']*(data['train_lift']+data['train_strength']+data['train_supp'])/15 # measure of the training volume
data['age_dec'] = data['age']/10 # compute the age (in decades)
data['age_start_dec'] = data['age_start']/10 # compute the start age (in decades)
data['sport0'] = data[['sport0_power','sport0_body','sport0_cf','sport0_ball','sport0_fit','sport0_endure','sport0_track',\
    'sport0_ma','sport0_yoga','sport0_gym','sport0_strength','sport0_impact']].max(axis=1) # prior sport
data['pa'] = data[['pa_power','pa_body','pa_cf','pa_ball','pa_fit','pa_endure','pa_track','pa_ma','pa_yoga']].max(axis=1) # concurrent sport

# Select response variable and covariates
X = data[['sex', 'OA', 'age_dec', 'age_start_dec', 'nutrition','train_total', 'pown', 'pa','sport0']]
X = sm.add_constant(X)  # Adds a constant (intercept) term to the model
y = data['wrist'] # Select type of injury

#######Random tree fit##################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Initialize a decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=4)
# Fit the classifier
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
r = recall_score(y_test, y_pred)
print(f"Recall: {r:.4f}")
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# Print the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Injury', 'Injury'], rounded=True)
plt.show()

importances = clf.feature_importances_
for feature, imp in zip(X.columns, importances):
    print(f"{feature}: {imp:.4f}")

# Combine feature names and their importance scores
features = list(X.columns)
feature_importance = dict(zip(features, importances))

# Sort the features by importance
sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
plt.bar([item[0] for item in sorted_feature_importance], [item[1] for item in sorted_feature_importance])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance from Decision Tree')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
