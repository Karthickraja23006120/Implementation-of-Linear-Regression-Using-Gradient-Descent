# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preprocessing: Load data from a CSV file (50_Startups.csv) using pandas and separate features (X) and target values (y). Convert these values to floating-point numbers for scaling, then apply standard scaling to normalize both features and target values.

2.Add Bias Term to Features: In the linear_regression function, concatenate a column of ones to X1 to add a bias term, creating a new feature matrix, X, with an intercept term for the linear regression model.

3.Initialize Parameters and Gradient Descent: Initialize theta (parameter vector) to zeros. For a specified number of iterations, perform gradient descent by calculating predictions, computing the error, and updating theta using the learning rate and gradient.

4.Model Training: Update theta iteratively to minimize the cost function until the specified number of iterations is reached. This process optimizes theta to fit the scaled feature matrix and target values.

5.Prediction with New Data: After training, apply the model to new data. Scale the new input data, append a bias term, and make a prediction by taking the dot product with theta. Finally, transform the scaled prediction back to the original scale for interpretation.
## Program:

```

Program to implement the linear regression using gradient descent.
Developed by: Karthick Raja K
RegisterNumber:  212223240066

import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:

![Screenshot 2024-08-31 140350](https://github.com/user-attachments/assets/18385998-acfe-4b9f-a34a-6db9f81e7c00)

![Screenshot 2024-08-31 140603](https://github.com/user-attachments/assets/5d8fadbb-67e8-4c39-b4b9-bf74b3b70e19)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
