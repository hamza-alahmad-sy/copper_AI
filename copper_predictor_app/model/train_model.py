#import the necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.preprocessing import StandardScaler
import pickle
#load the dataset
DATA_FILE = Path(__file__).resolve().parent.parent / 'data' / 'copper_prediction_dataset_1000.csv'
Model_FILE = Path(__file__).resolve().parent.parent / 'model' / 'copper_model.pkl'
df = pd.read_csv(DATA_FILE)

#the features variable
feature_columns = ['global_demand_index','oil_price','usd_index',
                   'china_industry_output','energy_cost_index',
                   'market_sentiment','supply_disruption_index']
x=df[feature_columns]
y=df['copper_price']   #the target variable

#split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#scale the features
scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test)

#initialize and train the model
model = LinearRegression()
model.fit(x_train_scaled, y_train)
print("Model training finished.")

#make predictions on the test set
y_pred = model.predict(x_test_scaled)


#evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)

#show model coefficients
coefficients = model.coef_
intercept = model.intercept_
for feature_name, coef in zip(feature_columns, coefficients):
    print(f"{feature_name}: {coef}")
print(f"Intercept:", intercept)

#save the trained model to a file
with open(Model_FILE, 'wb') as file:
     pickle.dump((model, scaler), file)
print(f"Trained model saved to", Model_FILE)
