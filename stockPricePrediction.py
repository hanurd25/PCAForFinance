import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt


# Step 2: Load the dataset (S&P 500 data)
stock_data = yf.download('AAPL', start='2024-08-08', end='2026-01-01')
# Step 3: Feature Engineering - adding technical indicators
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['Volatility'] = stock_data['Close'].rolling(window=20).std()
stock_data.dropna(inplace=True)
# Step 4: Data Preprocessing - Scaling the data
features = stock_data[['Close', 'SMA_20', 'SMA_50', 'Volatility']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
# Step 5: Apply PCA
pca = PCA(n_components=2)  # Retaining 2 components
principal_components = pca.fit_transform(scaled_features)
# Step 6: Visualizing Explained Variance
explained_variance = pca.explained_variance_ratio_
plt.bar(range(len(explained_variance)), explained_variance)
plt.title('Explained Variance by Principal Components')
plt.show()
# Step 7: Building the Predictive Model
X = principal_components
y = stock_data['Close']  # Using 'Close' price as the target
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Using Linear Regression for simplicity
model = LinearRegression()
model.fit(X_train, y_train)
# Step 8: Evaluating the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Step 9: Performance Comparison - Model without PCA
model_no_pca = LinearRegression()
model_no_pca.fit(X_train, y_train)
y_pred_no_pca = model_no_pca.predict(X_test)
mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)
print(f'MSE without PCA: {mse_no_pca}')
# Step 10: Compare the Results
print(f'Performance with PCA: {mse}')
print(f'Performance without PCA: {mse_no_pca}')
