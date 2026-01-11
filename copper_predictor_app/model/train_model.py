
from pathlib import Path 
import pandas as pd 
import numpy as np 
from sklearn .model_selection import train_test_split 
from sklearn .linear_model import LinearRegression 
from sklearn .metrics import r2_score ,mean_absolute_error ,mean_squared_error 
from sklearn .preprocessing import StandardScaler 
import pickle 
import sqlite3 


DB_FILE =Path (__file__ ).resolve ().parent .parent /'data'/'copper_data.db'
Model_FILE =Path (__file__ ).resolve ().parent .parent /'model'/'copper_model.pkl'


conn =sqlite3 .connect (DB_FILE )
df =pd .read_sql_query ("SELECT * FROM copper_data",conn )
conn .close ()


feature_columns =['global_demand_index','oil_price','usd_index',
'china_industry_output','energy_cost_index',
'market_sentiment','supply_disruption_index','copper_price']
x =df [feature_columns ]
y =df ['Next_Day_Copper_Price']


x_train ,x_test ,y_train ,y_test =train_test_split (x ,y ,test_size =0.2 ,random_state =42 )


scaler =StandardScaler ()
x_train_scaled =scaler .fit_transform (x_train )
x_test_scaled =scaler .transform (x_test )


model =LinearRegression ()
model .fit (x_train_scaled ,y_train )
print ("Model training finished.")


y_pred =model .predict (x_test_scaled )



mae =mean_absolute_error (y_test ,y_pred )
mse =mean_squared_error (y_test ,y_pred )
rmse =np .sqrt (mse )
mape =np .mean (np .abs ((y_test -y_pred )/y_test ))*100 

print ("MAE:",mae )
print ("MSE:",mse )
print ("RMSE:",rmse )
print ("MAPE:",mape )


coefficients =model .coef_ 
intercept =model .intercept_ 
for feature_name ,coef in zip (feature_columns ,coefficients ):
    print (f"{feature_name }: {coef }")
print (f"Intercept:",intercept )


with open (Model_FILE ,'wb')as file :
     pickle .dump ((model ,scaler ),file )
print (f"Trained model saved to",Model_FILE )
