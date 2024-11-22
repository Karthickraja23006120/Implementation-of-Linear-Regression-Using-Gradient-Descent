# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method. 
3. Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7. Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Karthick Raja K
RegisterNumber: 212223240066
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for c in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('/content/50_Startups.csv',header=None)
x=(data.iloc[1:, :-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")

```

## Output:
### Profit Prediction Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d64e3ca6-c94d-49b5-9116-a817b1d6d623)

![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/92625829-e1c6-473f-8f6a-f00d6209bdd6)
### Compute Cost Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/9e384697-fc9f-4277-92c1-841b285cd101)
### h(x) Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d8b272d5-104d-4cdb-942c-b849e8b54300)
### Cost function using Gradient Descent Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/c1ac8e0b-f252-4aac-8984-2c9f00da624a)
### Profit for the Population 35,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/61aad46d-b2d2-47d7-a7d7-ece05043cf30)
### Profit for the Population 70,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/70f9f953-a1da-4225-be06-19b89e9b42fe)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python
