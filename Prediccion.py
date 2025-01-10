import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# ...existing code...

# Load the training dataset
df_train = pd.read_csv('fraudtrain.csv')

# Load the testing dataset
df_test = pd.read_csv('fraudtest.csv')

# Print the column names
print(df_train.columns)

# Comment out the data visualization code
# Create a histogram for 'amt'
# plt.figure(figsize=(10, 6))
# df_train['amt'].hist(bins=50, color='blue')
# plt.title('Distribution of Amount')
# plt.xlabel('Amount')
# plt.ylabel('Frequency')
# plt.show()

# Generate a heatmap for feature correlations
# plt.figure(figsize=(12, 8))
# numeric_df = df_train.select_dtypes(include=['float64', 'int64'])
# correlation_matrix = numeric_df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Feature Correlation Heatmap')
# plt.show()

# Plot total transaction amount over time
# df_train['trans_date_trans_time'] = pd.to_datetime(df_train['trans_date_trans_time'])
# df_train.set_index('trans_date_trans_time', inplace=True)
# daily_transaction_amount = df_train['amt'].resample('D').sum()

# plt.figure(figsize=(14, 7))
# daily_transaction_amount.plot()
# plt.title('Total Transaction Amount Over Time')
# plt.xlabel('Date')
# plt.ylabel('Total Transaction Amount')
# plt.show()

# Create a histogram for 'unix_time'
# plt.figure(figsize=(10, 6))
# df_train['unix_time'].hist(bins=50, color='green')
# plt.title('Distribution of Unix Time')
# plt.xlabel('Unix Time')
# plt.ylabel('Frequency')
# plt.show()


# Generate a bar plot for 'is_fraud'
# plt.figure(figsize=(8, 6))
# sns.countplot(x='is_fraud', hue='is_fraud', data=df_train, palette='viridis', legend=False)
# plt.title('Distribution of Fraudulent Transactions')
# plt.xlabel('Is Fraud')
# plt.ylabel('Count')
# plt.show()

# Prepare the data for regression and classification
X_train = df_train[['amt', 'lat', 'long']]
y_train_reg = df_train['amt']
y_train_clf = df_train['is_fraud']

X_test = df_test[['amt', 'lat', 'long']]
y_test_reg = df_test['amt']
y_test_clf = df_test['is_fraud']

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_reg)
y_pred_reg = lin_reg.predict(X_test)

print("Linear Regression:")
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("R2:", r2_score(y_test_reg, y_pred_reg))

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_clf)
y_pred_clf = log_reg.predict(X_test)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("MSE:", mean_squared_error(y_test_clf, y_pred_clf))
print("MAE:", mean_absolute_error(y_test_clf, y_pred_clf))
print("R2:", r2_score(y_test_clf, y_pred_clf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf, zero_division=1))

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X_train, y_train_clf)
y_pred_rf = rf_clf.predict(X_test)

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
print("MSE:", mean_squared_error(y_test_clf, y_pred_rf))
print("MAE:", mean_absolute_error(y_test_clf, y_pred_rf))
print("R2:", r2_score(y_test_clf, y_pred_rf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_rf, zero_division=1))

# XGBoost
xgb_clf = XGBClassifier(n_estimators=10, eval_metric='logloss')
xgb_clf.fit(X_train, y_train_clf)
y_pred_xgb = xgb_clf.predict(X_test)

print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_xgb))
print("MSE:", mean_squared_error(y_test_clf, y_pred_xgb))
print("MAE:", mean_absolute_error(y_test_clf, y_pred_xgb))
print("R2:", r2_score(y_test_clf, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_xgb, zero_division=1))

import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'results_df' is your results DataFrame
results = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [accuracy_score(y_test_clf, y_pred_clf), accuracy_score(y_test_clf, y_pred_rf), accuracy_score(y_test_clf, y_pred_xgb)],
    'MSE': [mean_squared_error(y_test_clf, y_pred_clf), mean_squared_error(y_test_clf, y_pred_rf), mean_squared_error(y_test_clf, y_pred_xgb)],
    'MAE': [mean_absolute_error(y_test_clf, y_pred_clf), mean_absolute_error(y_test_clf, y_pred_rf), mean_absolute_error(y_test_clf, y_pred_xgb)],
    'R2': [r2_score(y_test_clf, y_pred_clf), r2_score(y_test_clf, y_pred_rf), r2_score(y_test_clf, y_pred_xgb)]
}

results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)

# Create subplots for each metric to visualize them separately
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Accuracy bar plot
results_df['Accuracy'].plot(kind='bar', ax=ax[0, 0], color='skyblue', legend=False)
ax[0, 0].set_title('Model Accuracy')
ax[0, 0].set_ylabel('Accuracy')
ax[0, 0].set_ylim(0, 1)  # Accuracy range
ax[0, 0].set_xticklabels(results_df.index, rotation=45)
for p in ax[0, 0].patches:
    ax[0, 0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# MSE bar plot
results_df['MSE'].plot(kind='bar', ax=ax[0, 1], color='lightcoral', legend=False)
ax[0, 1].set_title('Mean Squared Error (MSE)')
ax[0, 1].set_ylabel('MSE')
ax[0, 1].set_xticklabels(results_df.index, rotation=45)
for p in ax[0, 1].patches:
    ax[0, 1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# MAE bar plot
results_df['MAE'].plot(kind='bar', ax=ax[1, 0], color='lightgreen', legend=False)
ax[1, 0].set_title('Mean Absolute Error (MAE)')
ax[1, 0].set_ylabel('MAE')
ax[1, 0].set_xticklabels(results_df.index, rotation=45)
for p in ax[1, 0].patches:
    ax[1, 0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# R2 bar plot
results_df['R2'].plot(kind='bar', ax=ax[1, 1], color='gold', legend=False)
ax[1, 1].set_title('R2 Score')
ax[1, 1].set_ylabel('R2')
ax[1, 1].set_ylim(-1, 1)  # R2 range
ax[1, 1].set_xticklabels(results_df.index, rotation=45)
for p in ax[1, 1].patches:
    ax[1, 1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Adjust layout and show plot
plt.tight_layout()
plt.show()




