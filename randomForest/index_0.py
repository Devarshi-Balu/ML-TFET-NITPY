import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
train_data = pd.read_excel('Complete Dataset.xlsx')

# Ensure no negative values in Id and handle very small values
train_data['Id'] = np.where(train_data['Id'] < 1e-18, 1e-18, train_data['Id'])
train_data['Log_Id'] = np.log10(train_data['Id'])

# Separate inputs and output
X = train_data[['Tsi', 'Tox', 'Nc','Nd', 'Ns', 'Vds', 'Vgs']]
y = train_data['Log_Id']

# Polynomial features and scaling
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
test_score = model.score(X_test, y_test)
print(f"Random Forest Test Score: {test_score}")

# Load TCAD data
tcad_data = pd.read_csv('Complete Test.csv')
tcad_data['Id'] = np.where(tcad_data['Id'] < 1e-18, 1e-18, tcad_data['Id'])

# Select input parameters
Tsi, Tox, Nd, Ns, Vds, max_vgs = 5, 2, 1e19, 1e20, 0.1, 2.0  # Example values

vgs_values = np.linspace(0, max_vgs, 100)
input_data = pd.DataFrame({'Tsi': [Tsi] * len(vgs_values), 'Tox': [Tox] * len(vgs_values),
                           'Nd': [Nd] * len(vgs_values), 'Ns': [Ns] * len(vgs_values),
                           'Vds': [Vds] * len(vgs_values), 'Vgs': vgs_values})

# Prepare data for prediction
input_poly = poly.transform(input_data)
input_scaled = scaler.transform(input_poly)
predicted_log_id = model.predict(input_scaled)
predicted_id = np.maximum(np.power(10, predicted_log_id), 1e-18)

# Plot the results
plt.plot(vgs_values, predicted_id, label='Predicted (Random Forest)')
plt.xlabel('Vgs (V)')
plt.ylabel('Id (A)')
plt.title('Vgs vs Id (Linear Scale) - Random Forest')
plt.grid(True)
plt.legend()
plt.show()
