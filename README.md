# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.

2. Type the required program.

3. Print the program.

4. End the program. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Roghith.k
RegisterNumber:212222040135
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![ex 3 1.profit prediction](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/7ebf08fc-cf45-4adc-9877-6b0c99ba8edc)
![ex 3 2.function output](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/d9ebe72b-f7f9-4606-b0c1-2998b992e951)
![ex 3 3.Gradient Descent](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/487ed6d6-2896-47ce-a86c-af10dc81c9c0)
![ex 3 4.Cost function using gradient descent](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/627ca95f-32b8-49e6-ba01-7acc5ffcf02b)
![ex 3 5.Linear regression using profit prediction](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/eb180656-c5af-4aae-9ffd-f1353b9fef37)
![ex 3 6.Profit prediction for a population of 35,000](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/70287ffe-78a8-416f-b330-7a526a3421d9)
![ex 3 7.Profit prediction for a population of 70,000](https://github.com/RoghithKrishnamoorthy/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119475474/de0adb8b-acf0-4b1a-91a3-447bebfb0bc7)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
