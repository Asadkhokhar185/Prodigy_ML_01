import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Training and Testing Datasets
train_data = pd.read_csv('C:/Users/PMLS/Downloads/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('C:/Users/PMLS/Downloads/house-prices-advanced-regression-techniques/test.csv')

# Define the features and target variable
features = ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 
            'BedroomAbvGr', 'FullBath', 'HalfBath', 'BsmtFullBath', 
            'BsmtHalfBath', 'YearBuilt', 'OverallQual', 'OverallCond', 
            'LotArea']
target = 'SalePrice'

# Prepare the training data
X_train = train_data[features]
y_train = train_data[target]

# Prepare the test data (only features)
X_test = test_data[features]

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Make Predictions on the Test Data
test_predictions = model.predict(X_test_imputed)

# Create a DataFrame to store the results (optional)
results = pd.DataFrame({
    'Id': test_data['Id'],  # Assuming your test data has an 'Id' column
    'SalePrice': test_predictions
})

# Save the predictions to a CSV file (optional)
results.to_csv('test_predictions.csv', index=False)

# (Optional) Visualize the distribution of predicted prices
plt.figure(figsize=(10,6))
sns.histplot(test_predictions, kde=True)
plt.xlabel('Predicted Sale Price')
plt.title('Distribution of Predicted House Prices')
plt.show()
